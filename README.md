# [TIFS 2026] BGM: Background Mixup for X-Ray Prohibited Items Detection

Implementation of BGM, Background Mixup for X-Ray Prohibited Items Detection.


## Checkpoints

| Model | Dataset | Download |
| --- | --- | --- |
| Baseline | PIDray | HF Hub (coming soon) |
| BGM | PIDray | HF Hub (coming soon) |

Place each checkpoint as `model_final.pth` in `output/mr_detr_align_r50_12ep/` (Baseline) or `output/mr_detr_align_r50_12ep_bgm/` (BGM).

## Results

Recorded bbox AP on PIDray:

| Model | Positive | Easy | Hard | Hidden |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 74.8 | 78.9 | 74.3 | 52.7 |
| BGM | **76.8** | **80.4** | **76.3** | **56.1** |

## Evaluation

Evaluate both checkpoints on Positive:

```bash
NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash run_test.sh
```

Evaluate both checkpoints on Easy, Hard and Hidden:

```bash
NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash run_test_occlusion.sh
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{liu2026bgm,
  title={Bgm: Background mixup for x-ray prohibited items detection},
  author={Liu, Weizhe and Tao, Renshuai and Zhu, Hongguang and Sun, Yunda and Zhao, Yao and Wei, Yunchao},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgment

This implementation is based on Mr. DETR / detrex. We thank the authors for their contributions.
