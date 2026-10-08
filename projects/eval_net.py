#!/usr/bin/env python
"""Evaluate a checkpoint with the Mr. DETR framework."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detrex.modeling import ema


def main(args):
    args.eval_only = True
    cfg = LazyConfig.apply_overrides(LazyConfig.load(args.config_file), args.opts)
    if not cfg.train.init_checkpoint or not os.path.isfile(cfg.train.init_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.train.init_checkpoint}")
    default_setup(cfg, args)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)
    result = inference_on_dataset(model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator))
    print_csv_format(result)
    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
