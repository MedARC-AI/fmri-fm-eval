# This source code is licensed under the Apache License, Version 2.0

import argparse
import datetime
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import torch
import torch.nn as nn
import wandb
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import fmri_fm_eval.utils as ut
import fmri_fm_eval.version
from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import create_dataset, list_datasets
from fmri_fm_eval.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_logistic.yaml"

METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
}

# sklearn scoring names for LogisticRegressionCV
SKLEARN_SCORING = {
    "acc": "accuracy",
    "f1": "f1_macro",
}


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed logistic eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = f"{args.name_prefix}/{args.model}/{args.representation}/{args.dataset}"
    args.output_dir = f"{args.output_root}/{args.name}"
    output_dir = Path(args.output_dir)

    # remote backup location
    if args.remote_root:
        args.remote_dir = f"{args.remote_root}/{args.name}"
        if S3Path(args.remote_dir).exists():
            ut.rsync(args.remote_dir, args.output_dir)
    else:
        args.remote_dir = None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model logistic probe eval")
    print(f"version: {fmri_fm_eval.version.__version__}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating frozen backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))
    backbone.requires_grad_(False)
    backbone.to(device)
    print(f"backbone:\n{backbone}")

    # dataset
    print(f"creating dataset: {args.dataset} ({backbone.__space__})")
    dataset_dict = create_dataset(
        args.dataset, space=backbone.__space__, **(args.dataset_kwargs or {})
    )
    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")
    train_dataset: HFDataset = dataset_dict["train"]
    args.num_classes = train_dataset.num_classes

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    # extract features
    print("extracting features for all splits")
    start_time = time.monotonic()
    features_dict, targets_dict = extract_features(args, backbone, dataset_dict, device)
    extract_time = time.monotonic() - start_time
    print(f"feature extraction time: {datetime.timedelta(seconds=int(extract_time))}")

    for split, features in features_dict.items():
        print(f"{split} features: {features.shape}")

    # standardize features
    print("fitting StandardScaler on training features")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features_dict["train"])
    features_dict["train"] = train_features

    for split in features_dict:
        if split != "train":
            features_dict[split] = scaler.transform(features_dict[split])

    # fit logistic regression with cross-validation
    print(f"fitting LogisticRegressionCV (cv={args.cv_folds}, Cs={args.Cs})")
    class_weight = "balanced" if args.balanced_sampling else None
    scoring = SKLEARN_SCORING.get(args.cv_metric, args.cv_metric)

    clf = LogisticRegressionCV(
        Cs=args.Cs,
        cv=args.cv_folds,
        scoring=scoring,
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.seed,
        n_jobs=args.num_workers,
    )

    fit_start = time.monotonic()
    clf.fit(features_dict["train"], targets_dict["train"])
    fit_time = time.monotonic() - fit_start
    print(f"fit time: {datetime.timedelta(seconds=int(fit_time))}")
    print(f"best C: {clf.C_[0]:.4g}")

    # evaluate on all splits
    print("evaluating on all splits")
    header = {
        "model": args.model,
        "repr": args.representation,
        "clf": "logistic",
        "dataset": args.dataset,
        "C": float(clf.C_[0]),
    }
    eval_stats = {
        "eval/C_best": float(clf.C_[0]),
    }
    table = []
    log_wandb = args.wandb and ut.is_main_process()

    for split in features_dict:
        features = features_dict[split]
        targets = targets_dict[split]

        preds = clf.predict(features)
        record = {**header, "split": split}

        bootstrap_result = bootstrap_ci(args, preds, targets)

        for metric in args.metrics:
            metric_fn = METRICS[metric]
            score = metric_fn(targets, preds)
            record[metric] = eval_stats[f"eval/{split}/{metric}"] = score
            record[f"{metric}_std"] = bootstrap_result[metric]["std"]

        table.append(record)
        np.savez(output_dir / f"preds_{split}.npz", preds=preds, targets=targets)

    table = pd.DataFrame.from_records(table)
    table_fmt = table.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results:\n\n{table_fmt}\n\n")
    table.to_csv(output_dir / "eval_table.csv", index=False)

    with (output_dir / "eval_log.json").open("w") as f:
        print(json.dumps(eval_stats), file=f)

    if log_wandb:
        wandb.log(eval_stats)

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)


@torch.inference_mode()
def extract_features(
    args: DictConfig,
    backbone: nn.Module,
    dataset_dict: dict[str, HFDataset],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    backbone.eval()
    print_freq = args.get("print_freq", 20)

    features_dict = {}
    targets_dict = {}

    for split, dataset in dataset_dict.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers else None,
            drop_last=False,
        )

        metric_logger = ut.MetricLogger(delimiter="  ")
        header = f"extract ({split})"

        all_features = []
        all_targets = []

        for batch in metric_logger.log_every(loader, print_freq, header, len(loader)):
            batch = ut.send_data(batch, device)
            target = batch.pop("target")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                cls_embeds, reg_embeds, patch_embeds = backbone(batch)

            all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
            embeds = all_embeds[args.representation]

            # average over sequence dimension: (n, l, d) -> (n, d)
            if embeds.ndim == 3:
                embeds = embeds.mean(dim=1)

            all_features.append(embeds.cpu().float().numpy())
            all_targets.append(target.cpu().numpy())

        features_dict[split] = np.concatenate(all_features, axis=0)
        targets_dict[split] = np.concatenate(all_targets, axis=0)

    return features_dict, targets_dict


def bootstrap_ci(args: DictConfig, preds: np.ndarray, targets: np.ndarray):
    random_state = sklearn.utils.check_random_state(args.seed)

    sample_scores = defaultdict(list)
    for _ in range(500):
        preds_, targets_ = sklearn.utils.resample(
            preds, targets, random_state=random_state, stratify=targets
        )
        for metric in args.metrics:
            metric_fn = METRICS[metric]
            sample_scores[metric].append(metric_fn(targets_, preds_))

    result = {}
    for metric, values in sample_scores.items():
        result[metric] = {"mean": np.mean(values), "std": np.std(values)}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "dataset",
        type=str,
        help=f"[{', '.join(list_datasets())}]",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model = args.model
    cfg.representation = args.representation
    cfg.dataset = args.dataset
    main(cfg)
