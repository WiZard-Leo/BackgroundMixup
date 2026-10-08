from detectron2.config import LazyConfig
from .models.deformable_detr_r50 import model

train = LazyConfig.load_rel("../../../configs/common/eval_runtime.py", "train")
dataloader = LazyConfig.load_rel("../../../configs/common/data/pidray_occlusion_eval.py", "dataloader")
model.with_box_refine = True
model.as_two_stage = True
model.num_classes = 12
model.criterion.num_classes = 12
model.transformer.encoder.feedforward_dim = 2048
model.transformer.decoder.feedforward_dim = 2048
model.transformer.encoder.attn_dropout = 0.0
model.transformer.encoder.ffn_dropout = 0.0
model.transformer.decoder.attn_dropout = 0.0
model.transformer.decoder.ffn_dropout = 0.0
model.mixed_selection = True
model.transformer.mixed_selection = True
model.transformer.decoder.look_forward_twice = True
model.select_box_nums_for_evaluation = 300
model.device = "${train.device}"
dataloader.evaluator.output_dir = "${train.output_dir}"
