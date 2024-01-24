# Codes contributed by Zijian Kang: Copyright (c) Megvii Inc. and its affiliates. All Rights Reserved
# Codes contributed by detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import time

from detectron2.config import CfgNode as CN
from detectron2.utils.events import EventWriter, get_event_storage
import numpy as np
import torch.utils.data

import torch.nn as nn

from detectron2.utils.env import seed_all_rng

from detectron2.evaluation.evaluator import *
from detectron2.solver.build import maybe_add_gradient_clipping

"""
This file contains the default logic to build a dataloader for training or testing.
"""


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def my_build_optimizer(cfg, model):
    params: list[dict[str, any]] = []
    memo: set[torch.nn.parameter.Parameter] = set()
    if isinstance(model, nn.Module):
        model = [model]

    custom_decay = False
    try:
        bb = cfg.MODEL.BACKBONE.NAME
    except:
        print('cfg.MODEL.BACKBONE.NAME not found! trying cfg.BACKBONE.NAME... (cfg here is cfg.MODEL.DISTILLER)')
        bb = cfg.BACKBONE_NAME
    
    if bb == "build_vitdet_backbone":
        custom_decay = True
        from model_zoo import get_vit_lr_decay_rate
        from functools import partial
        wt_decay_fn = partial(get_vit_lr_decay_rate, num_layers=int(cfg.MODEL.VIT.DEPTH), lr_decay_rate=cfg.SOLVER.WEIGHT_DECAY)
    
    for m in model:
        for key, value in m.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR

            # weight_decay = cfg.SOLVER.WEIGHT_DECAY
            weight_decay = cfg.SOLVER.WEIGHT_DECAY if not custom_decay else wt_decay_fn(key)
            # if "backbone" in key or "encoder" in key:
            #     lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr,
                        "weight_decay": weight_decay}]

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif optimizer_type == "ADAMW":
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


class AllMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, logger, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logger
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(
                1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds,
                               smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * \
                    (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter} {log}  {time}{data_time}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                log="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(20))
                        for k, v in storage.histories().items()
                        if k not in ['time', 'data_time', 'eta_seconds']
                    ]
                ),
                time="time: {:.4f}  ".format(
                    iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(
                    data_time) if data_time is not None else "",
                memory="max_mem: {:.0f}M".format(
                    max_mem_mb) if max_mem_mb is not None else "",
            )
        )
