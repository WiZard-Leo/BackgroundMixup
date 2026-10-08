import os

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper


dataloader = OmegaConf.create()

_split = os.environ.get("PIDRAY_TEST_SPLIT", "easy").lower()
_splits = {
    "positive": ("datasets/PIDray/annotations/test_positive.json", "datasets/PIDray/test"),
    "easy": (
        "datasets/PIDray/annotations/test_easy.json",
        "datasets/PIDray/test",
    ),
    "hard": (
        "datasets/PIDray/annotations/test_hard.json",
        "datasets/PIDray/test",
    ),
    "hidden": (
        "datasets/PIDray/annotations/test_hidden.json",
        "datasets/PIDray/test",
    ),
}

if _split not in _splits:
    raise ValueError(
        f"Unsupported PIDRAY_TEST_SPLIT={_split!r}. "
        f"Expected one of {sorted(_splits)}."
    )

_dataset_name = f"pidray_test_occlusion_{_split}"
_json_file, _image_root = _splits[_split]
_dataset_root = os.environ.get("DETECTRON2_DATASETS", "datasets")
_json_file = os.path.join(_dataset_root, os.path.relpath(_json_file, "datasets"))
_image_root = os.path.join(_dataset_root, os.path.relpath(_image_root, "datasets"))

register_coco_instances(_dataset_name, {}, _json_file, _image_root)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names=_dataset_name, filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
