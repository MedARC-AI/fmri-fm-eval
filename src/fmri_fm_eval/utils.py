# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/utils.py
# beit3: https://github.com/microsoft/unilm/blob/master/beit3/utils.py
# capi: https://github.com/facebookresearch/capi/blob/main/utils.py
# dinov2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/param_groups.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/cuda.py
# dino: https://github.com/facebookresearch/dino/blob/main/utils.py

import datetime
import fnmatch
import inspect
import math
import os
import random
import subprocess
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist


# these very useful utils copied from deit with only minor changes
# thanks to the original authors, wherever you are


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        value = float(value)
        if math.isfinite(value):
            self.deque.append(value)
            self.count += n
            self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.count:
            return float("nan")
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if not self.count:
            return float("nan")
        return self.total / self.count

    @property
    def max(self):
        if not self.count:
            return float("nan")
        return max(self.deque)

    @property
    def value(self):
        if not self.count:
            return float("nan")
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, (torch.Tensor, np.generic)):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, total_steps=None):
        i = 0
        total_steps = total_steps or len(iterable)
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(total_steps))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            if i >= total_steps:
                break
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_steps - 1:
                eta_seconds = iter_time.global_avg * (total_steps - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            total_steps,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / total_steps
            )
        )


def setup_for_distributed(log_path=None):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    is_master = is_main_process()

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            # tee to log file
            if log_path and "file" not in kwargs:
                with open(log_path, "a") as f:
                    builtin_print(*args, file=f, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # removed slurm block, can add if we use slurm
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank})")
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=args.rank,
        device_id=args.gpu,
    )
    torch.distributed.barrier()


def rsync(src_path: str, dst_path: str):
    src_path = str(src_path)
    dst_path = str(dst_path)
    assert all(urlparse(p).scheme in {"", "s3"} for p in [src_path, dst_path]), (
        "only local and s3 paths supported"
    )

    cmd = ["aws", "s3", "sync", src_path, dst_path]
    subprocess.run(cmd, check=True)


# optimization utils


def get_param_groups(model):
    param_groups = {}

    no_weight_decay = getattr(model, "__no_decay_params__", [])

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr_multiplier = wd_multiplier = 1.0

        if name.endswith(".bias") or "norm" in name:
            wd_multiplier = 0.0

        if any(fnmatch.fnmatch(name, pat) for pat in no_weight_decay):
            wd_multiplier = 0.0

        key = (lr_multiplier, wd_multiplier)

        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr_multiplier": lr_multiplier,
                "wd_multiplier": wd_multiplier,
            }

        param_groups[key]["params"].append(param)

    param_groups = list(param_groups.values())
    return param_groups


def update_lr(param_groups, lr: float):
    # cast to float or else np scalars corrupt checkpoint
    lr = float(lr)
    for group in param_groups:
        group["lr"] = lr * group["lr_multiplier"]


def update_wd(param_groups, weight_decay: float):
    # cast to float or else np scalars corrupt checkpoint
    weight_decay = float(weight_decay)
    for group in param_groups:
        group["weight_decay"] = weight_decay * group["wd_multiplier"]


# moving data to cuda utils copied from capi
# added device argument


def send_data(x, device=None):
    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if isinstance(x, torch.Tensor):
        x = x.to(device=device, non_blocking=True)
        if device.type == "cuda":
            x.record_stream(torch.cuda.current_stream(device))
        return x
    if isinstance(x, dict):
        return {k: send_data(v, device=device) for k, v in x.items()}
    if isinstance(x, list):
        return [send_data(v, device=device) for v in x]
    return x


def pre_send_to_cuda_wrapper(generator, device=None):
    """From apex"""
    data = None
    stream = torch.cuda.Stream(device)
    for next_data in generator:
        # Move to GPU
        with torch.cuda.stream(stream):
            next_data = send_data(next_data, device=device)
        if data is not None:
            yield data
        torch.cuda.current_stream(device).wait_stream(stream)
        data = next_data


def infinite_data_wrapper(loader, num_steps: int | None = None):
    step = 0
    iterator = iter(loader)
    while num_steps is None or step < num_steps:
        try:
            sample = next(iterator)
            yield sample
            step += 1
        except StopIteration:
            # update distributed sampler epoch if necessary
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "epoch"):
                loader.sampler.epoch += 1
            iterator = iter(loader)


# other misc utils


# from dino
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# from timm
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


# mine :)
def filter_kwargs(func, kwargs):
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs
