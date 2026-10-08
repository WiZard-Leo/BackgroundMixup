"""Evaluation runtime settings; train is the legacy detectron2 config key."""
train = dict(
    output_dir="./output/evaluation",
    init_checkpoint="",
    device="cuda",
    seed=-1,
    model_ema=dict(
        enabled=False,
        decay=0.999,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
)
