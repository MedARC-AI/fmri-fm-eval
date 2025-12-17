# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import argparse
import datetime
import json
import math
import time
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import wandb
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import fmri_fm_eval.utils as ut
from fmri_fm_eval.heads import (
    ClassifierGrid,
    LinearClassifier,
    AttnPoolClassifier,
    pool_representation,
)
from fmri_fm_eval.models.registry import create_model
from fmri_fm_eval.datasets.registry import create_dataset

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed probe eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

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

    print("fMRI foundation model probe eval")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"cwd: {Path.cwd()}")
    print(ut.get_sha())
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
    args.num_classes = dataset_dict["train"].__num_classes__
    args.task = dataset_dict["train"].__task__
    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")

    for split, ds in dataset_dict.items():
        ds.set_transform(transform)

    loaders_dict = {}
    for split, dataset in dataset_dict.items():
        loaders_dict[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
        )
    # we could also support more splits or different split names, but for now can keep
    # things simple.
    train_loader = loaders_dict["train"]
    val_loader = loaders_dict["validation"]
    test_loader = loaders_dict["test"]

    # prediction heads
    print("running backbone on example batch to get embedding shape")
    embed_shape = get_embedding_shape(backbone, args.representation, dataset_dict["train"], device)
    print(f"embedding feature shape: {args.representation}: {embed_shape}")

    print("initializing sweep of classifier heads")
    classifiers, param_groups = make_classifiers(args, embed_shape, args.num_classes)
    model = ClassifierGrid(backbone, args.representation, classifiers)
    model.to(device)
    print(f"classifiers:\n{model.classifiers}")
    num_params = sum(p.numel() for p in model.classifiers.parameters())
    num_params_train = sum(p.numel() for p in model.classifiers.parameters() if p.requires_grad)
    print(f"classifier params (train): {num_params / 1e6:.1f}M ({num_params_train / 1e6:.1f}M)")

    # optimizer
    print("setting up optimizer")
    total_batch_size = args.batch_size * args.accum_iter
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        print(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        print(f"lr: {args.lr:.2e}")

    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_epochs * args.steps_per_epoch
    lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps)
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # load checkpoint/resume training
    ckpt_meta = ut.load_model(args, model, optimizer)
    if ckpt_meta is not None:
        best_info = ckpt_meta["best_info"]
    else:
        best_info = {"loss": float("inf"), "hparam": None, "epoch": None}

    # training loss
    if args.task == "classification":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.task == "regression":
        criterion = nn.MSELoss(reduction="none")
    else:
        raise ValueError(f"Unknown task: {args.task}.")

    print(f"start training for {args.epochs} epochs")
    log_wandb = args.wandb and ut.is_main_process()
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            train_loader,
            optimizer,
            lr_schedule,
            epoch,
            device,
        )

        val_stats = evaluate(
            args,
            model,
            criterion,
            val_loader,
            epoch,
            device,
            eval_name="validation",
        )

        if log_wandb:
            wandb.log(val_stats, (epoch + 1) * args.steps_per_epoch)

        hparam, loss = get_best_hparams(model, val_stats)
        print(f"epoch: [{epoch}]  best hparam: {hparam}  best loss: {loss:.3f}")

        if loss < best_info["loss"]:
            best_info = {"loss": loss, "hparam": hparam, "epoch": epoch}
            is_best = True
        else:
            is_best = False

        hparam_fmt = format_hparam(model.hparam_id_map[hparam], hparam)
        best_stats = {
            "eval/validation/lr_scale_best": hparam[0],
            "eval/validation/wd_scale_best": hparam[1],
            "eval/validation/loss_best": val_stats[f"eval/validation/loss_{hparam_fmt}"],
        }
        if args.task == "classification":
            best_stats["eval/validation/acc1_best"] = val_stats[
                f"eval/validation/acc1_{hparam_fmt}"
            ]

        if log_wandb:
            wandb.log(best_stats, (epoch + 1) * args.steps_per_epoch)

        merged_stats = {"epoch": epoch, **train_stats, **val_stats, **best_stats}
        with (output_dir / "log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        ckpt_meta = {"best_info": best_info}
        ut.save_model(args, epoch, model, optimizer, meta=ckpt_meta, is_best=is_best)

        if args.remote_dir:
            print(f"backing up to remote: {args.remote_dir}")
            ut.rsync(args.remote_dir, output_dir)

    print("evaluating best model on test set")
    best_ckpt = torch.load(
        output_dir / "checkpoint-best.pth", map_location="cpu", weights_only=True
    )
    model.load_state_dict(best_ckpt["model"])
    best_info = best_ckpt["meta"]["best_info"]
    print(f"best model info:\n{json.dumps(best_info)}")

    test_stats = evaluate(
        args,
        model,
        criterion,
        test_loader,
        best_info["epoch"],
        device,
        eval_name="test",
    )

    hparam = best_info["hparam"]
    hparam_fmt = format_hparam(model.hparam_id_map[hparam], hparam)
    best_stats = {
        "eval/test/epoch_best": best_info["epoch"],
        "eval/test/lr_scale_best": hparam[0],
        "eval/test/wd_scale_best": hparam[1],
        "eval/test/loss_best": test_stats[f"eval/test/loss_{hparam_fmt}"],
    }
    if args.task == "classification":
        best_stats["eval/test/acc1_best"] = test_stats[f"eval/test/acc1_{hparam_fmt}"]

    print(f"best model test stats:\n{json.dumps(best_stats)}")
    with (output_dir / "test_log.json").open("a") as f:
        print(json.dumps(best_stats), file=f)

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)


@torch.inference_mode()
def get_embedding_shape(
    backbone: nn.Module,
    representation: str,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    loader = DataLoader(dataset, batch_size=1)
    example_batch = next(iter(loader))
    example_batch = ut.send_data(example_batch, device)

    cls_embeds, reg_embeds, patch_embeds = backbone(example_batch)
    pooled = pool_representation(
        cls_embeds, reg_embeds, patch_embeds, representation=representation
    )
    embed_shape = pooled.shape[1:]
    return embed_shape


def make_classifiers(
    args: DictConfig,
    embed_shape: tuple[int, ...],
    num_classes: int,
):
    # create sweep of classifier heads with varying hparams
    all_classifiers = {}
    param_groups = {}

    assert len(embed_shape) in {1, 2}

    if len(embed_shape) == 1:
        clf_fn = partial(LinearClassifier, embed_shape[-1], num_classes)
    else:
        clf_fn = partial(
            AttnPoolClassifier,
            embed_shape[-1],
            num_classes,
            embed_dim=args.get("attn_pool_embed_dim"),
        )

    # all classifiers get same init
    init_state = None

    for lr_multiplier, wd_multiplier in product(args.lr_scale_grid, args.wd_scale_grid):
        clf = clf_fn()
        if init_state is None:
            init_state = clf.state_dict()
        else:
            clf.load_state_dict(init_state)

        all_classifiers[(lr_multiplier, wd_multiplier)] = clf

        for name, param in clf.named_parameters():
            param_wd_multiplier = wd_multiplier

            if name.endswith(".bias") or "norm" in name:
                param_wd_multiplier = 0.0

            key = (lr_multiplier, param_wd_multiplier)
            if key not in param_groups:
                param_groups[key] = {
                    "params": [],
                    "lr_multiplier": lr_multiplier,
                    "wd_multiplier": param_wd_multiplier,
                }
            param_groups[key]["params"].append(param)

    param_groups = list(param_groups.values())
    return all_classifiers, param_groups


def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int):
    warmup = np.linspace(0.0, 1.0, warmup_steps)
    decay = np.cos(np.linspace(0, np.pi, max(total_steps - warmup_steps, 0)))
    decay = (decay + 1) / 2
    lr_schedule = base_lr * np.concatenate([warmup, decay])
    lr_schedule = lr_schedule[:total_steps]
    return lr_schedule


def train_one_epoch(
    args: DictConfig,
    model: ClassifierGrid,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
):
    model.train()
    use_cuda = device.type == "cuda"
    log_wandb = args.wandb and ut.is_main_process()
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = args.steps_per_epoch * args.accum_iter if not args.debug else 10

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"train: [{epoch}]"
    all_meters = defaultdict(ut.SmoothedValue)

    num_classifiers = len(model.classifiers)

    data_loader = ut.infinite_data_wrapper(data_loader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)

        global_step = epoch * args.steps_per_epoch + (batch_idx + 1) // args.accum_iter
        need_update = (batch_idx + 1) % args.accum_iter == 0
        if need_update:
            lr = lr_schedule[global_step - 1]
            ut.update_lr(optimizer.param_groups, lr)

        target = batch.pop("target")

        # handle single target regression
        # predictions are always shape [batch_size, num_targets, num_classifiers], so
        # need to match second dimension.
        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)

        # expand last dimension of target to match prediction
        # note that the num_classifiers dimension has to go at the end bc this is
        # what nn.CrossEntropyLoss expects.
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        # nb, no loss scaler. can add if needed.
        (loss / args.accum_iter).backward()

        if need_update:
            # grad clip per classifier separately
            all_grad = []
            for clf in model.classifiers:
                grad = nn.utils.clip_grad_norm_(clf.parameters(), args.clip_grad)
                all_grad.append(grad)
            total_grad = torch.stack(all_grad).norm()
            optimizer.step()
            optimizer.zero_grad()

        if need_update:
            log_metric_dict = {
                "lr": lr,
                "loss": loss_value,
                "grad": total_grad.item(),
            }
            metric_logger.update(**log_metric_dict)

            all_metric_dict = {}
            all_metric_dict.update(
                {
                    f"loss_{format_hparam(ii, hparam)}": all_loss[ii].item()
                    for ii, hparam in enumerate(model.hparams)
                }
            )
            all_metric_dict.update(
                {
                    f"grad_{format_hparam(ii, hparam)}": all_grad[ii].item()
                    for ii, hparam in enumerate(model.hparams)
                }
            )

            for k, v in all_metric_dict.items():
                all_meters[k].update(v)

            if log_wandb:
                wandb.log({f"train/{k}": v for k, v in log_metric_dict.items()}, global_step)
                wandb.log({f"train/{k}": v for k, v in all_metric_dict.items()}, global_step)

        if use_cuda:
            torch.cuda.synchronize()

    print(f"{header} Summary:", metric_logger)

    stats = {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({f"train/{k}": meter.global_avg for k, meter in all_meters.items()})
    return stats


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: ClassifierGrid,
    criterion: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
):
    model.eval()
    use_cuda = device.type == "cuda"
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = len(data_loader) if not args.debug else 10

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"eval ({eval_name}): [{epoch}]"

    num_classifiers = len(model.classifiers)

    preds = []
    targets = []

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)
        target = batch.pop("target")

        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch)

        preds.append(pred.cpu().float())
        targets.append(target.cpu())

        if use_cuda:
            torch.cuda.synchronize()

    # average loss and acc over the full eval dataset
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    total_loss = criterion(preds, targets)
    total_loss = total_loss.reshape(-1, num_classifiers).mean(dim=0).tolist()
    if args.task == "classification":
        total_acc1 = [
            ut.accuracy(preds[:, :, ii], targets[:, ii])[0].item() for ii in range(num_classifiers)
        ]

    stats = {}
    stats.update(
        {
            f"loss_{format_hparam(ii, hparam)}": total_loss[ii]
            for ii, hparam in enumerate(model.hparams)
        }
    )
    if args.task == "classification":
        stats.update(
            {
                f"acc1_{format_hparam(ii, hparam)}": total_acc1[ii]
                for ii, hparam in enumerate(model.hparams)
            }
        )

    stats = {f"eval/{eval_name}/{k}": v for k, v in stats.items()}

    # TODO: return preds and targets to compute metrics offline?
    return stats


def format_hparam(idx: int, hparam: tuple[float, float]) -> str:
    lr, weight_decay = hparam
    return f"{idx:03d}_lr{lr:.1e}_wd{weight_decay:.1e}"


def get_best_hparams(model: ClassifierGrid, stats: dict[str, float]):
    losses = [
        stats[f"eval/validation/loss_{format_hparam(ii, hparam)}"]
        for ii, hparam in enumerate(model.hparams)
    ]
    best_id = np.argmin(losses)
    best_hparam = model.hparams[best_id]
    best_loss = losses[best_id]
    return best_hparam, best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
