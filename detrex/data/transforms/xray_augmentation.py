# coding=utf-8

import cv2
import numpy as np
from fvcore.transforms.transform import NoOpTransform, Transform

from detectron2.data.transforms import Augmentation

__all__ = [
    "BgmCpm",
    "BgmCpmTransform",
    "BgmSpm",
    "BgmSpmTransform",
]


class BgmCpmTransform(Transform):
    """Apply pre-sampled Color Patch Mixup on image pixels only."""

    def __init__(self, patches, colors, alphas):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        output = img.copy()
        for patch, color, alpha in zip(self.patches, self.colors, self.alphas):
            px1, py1, px2, py2 = map(int, patch)
            if px2 <= px1 or py2 <= py1:
                continue
            patch_roi = output[py1:py2, px1:px2]
            color_patch = np.broadcast_to(color, patch_roi.shape).astype(patch_roi.dtype, copy=False)
            output[py1:py2, px1:px2] = cv2.addWeighted(
                patch_roi,
                1.0 - alpha,
                color_patch,
                alpha,
                0,
            )
        return output

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation


class BgmCpm(Augmentation):
    """Color Patch Mixup for X-ray images."""

    def __init__(
        self,
        n_patches=(1, 3),
        ratio=(0.02, 0.2),
        squared=False,
        mix_alpha=(0.3, 0.7),
        perform_prob=1.0,
    ):
        super().__init__()
        if isinstance(n_patches, int):
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, float):
            ratio = (ratio, ratio)
        if isinstance(mix_alpha, float):
            mix_alpha = (mix_alpha, mix_alpha)
        self._init(locals())

    def _sample_patch(self, height, width):
        if self.squared:
            sampled_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
            ratio_h, ratio_w = sampled_ratio, sampled_ratio
        else:
            ratio_h = np.random.uniform(self.ratio[0], self.ratio[1])
            ratio_w = np.random.uniform(self.ratio[0], self.ratio[1])

        ph = int(height * ratio_h)
        pw = int(width * ratio_w)
        ph = min(max(ph, 1), height)
        pw = min(max(pw, 1), width)

        max_x = width - pw
        max_y = height - ph
        px1 = 0 if max_x <= 0 else np.random.randint(0, max_x + 1)
        py1 = 0 if max_y <= 0 else np.random.randint(0, max_y + 1)
        return [px1, py1, px1 + pw, py1 + ph]

    def get_transform(self, image):
        if np.random.rand() > self.perform_prob:
            return NoOpTransform()

        height, width = image.shape[:2]
        num_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)

        patches = []
        colors = []
        alphas = []
        for _ in range(num_patches):
            patch = self._sample_patch(height, width)
            color = np.random.randint(0, 256, (1, 1, image.shape[2]), dtype=np.uint8).astype(
                np.float32
            )
            noise = np.random.poisson(color).astype(np.float32)
            color_with_noise = np.clip(color + noise, 0, 255).astype(np.uint8)
            alpha = float(np.random.uniform(self.mix_alpha[0], self.mix_alpha[1]))

            patches.append(patch)
            colors.append(color_with_noise)
            alphas.append(alpha)

        return BgmCpmTransform(np.asarray(patches), colors, alphas)


class BgmSpmTransform(Transform):
    """Apply pre-sampled Self Patch Mixup on image pixels only."""

    def __init__(self, patches_info, alphas):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        output = img.copy()
        for patch, alpha in zip(self.patches_info, self.alphas):
            ox1, oy1, ox2, oy2, tx1, ty1, tx2, ty2 = map(int, patch)
            if ox2 <= ox1 or oy2 <= oy1 or tx2 <= tx1 or ty2 <= ty1:
                continue

            source_patch = output[oy1:oy2, ox1:ox2].copy()
            target_roi = output[ty1:ty2, tx1:tx2]
            if source_patch.shape != target_roi.shape:
                continue

            output[ty1:ty2, tx1:tx2] = cv2.addWeighted(
                target_roi,
                1.0 - alpha,
                source_patch,
                alpha,
                0,
            )
        return output

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation


class BgmSpm(Augmentation):
    """Self Patch Mixup / Background Mixup for X-ray images."""

    def __init__(
        self,
        n_patches=(1, 3),
        ratio=(0.02, 0.2),
        squared=False,
        mix_alpha=(0.3, 0.7),
        perform_prob=1.0,
        img_format="RGB",
    ):
        super().__init__()
        if isinstance(n_patches, int):
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, float):
            ratio = (ratio, ratio)
        if isinstance(mix_alpha, float):
            mix_alpha = (mix_alpha, mix_alpha)
        self._init(locals())

    def _get_saliency_bbox(self, image):
        if self.img_format == "RGB":
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l_mean, a_mean, b_mean = np.mean(lab_image, axis=(0, 1))
        saliency_map = np.sqrt(
            (lab_image[:, :, 0] - l_mean) ** 2
            + (lab_image[:, :, 1] - a_mean) ** 2
            + (lab_image[:, :, 2] - b_mean) ** 2
        )
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        _, fg_mask = cv2.threshold(saliency_map.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, image.shape[1], image.shape[0]

        all_points = np.concatenate(contours, axis=0)
        x_min, y_min = np.min(all_points, axis=0)[0]
        x_max, y_max = np.max(all_points, axis=0)[0]
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def _sample_patch(self, height, width, valid_area):
        x_min, y_min, x_max, y_max = valid_area
        area_w = max(x_max - x_min, 1)
        area_h = max(y_max - y_min, 1)

        if self.squared:
            sampled_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
            pw = int(min(area_w, area_h) * sampled_ratio)
            ph = pw
        else:
            pw = int(area_w * np.random.uniform(self.ratio[0], self.ratio[1]))
            ph = int(area_h * np.random.uniform(self.ratio[0], self.ratio[1]))

        pw = min(max(pw, 1), area_w)
        ph = min(max(ph, 1), area_h)
        if pw >= area_w or ph >= area_h:
            return None

        orig_x1 = np.random.randint(x_min, x_max - pw + 1)
        orig_y1 = np.random.randint(y_min, y_max - ph + 1)

        max_tx = width - pw
        max_ty = height - ph
        target_x1 = 0 if max_tx <= 0 else np.random.randint(0, max_tx + 1)
        target_y1 = 0 if max_ty <= 0 else np.random.randint(0, max_ty + 1)

        return [
            orig_x1,
            orig_y1,
            orig_x1 + pw,
            orig_y1 + ph,
            target_x1,
            target_y1,
            target_x1 + pw,
            target_y1 + ph,
        ]

    def get_transform(self, image):
        if np.random.rand() > self.perform_prob:
            return NoOpTransform()

        height, width = image.shape[:2]
        valid_area = self._get_saliency_bbox(image)
        num_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)

        patches_info = []
        alphas = []
        for _ in range(num_patches):
            patch = self._sample_patch(height, width, valid_area)
            if patch is None:
                continue
            patches_info.append(patch)
            alphas.append(float(np.random.uniform(self.mix_alpha[0], self.mix_alpha[1])))

        if not patches_info:
            return NoOpTransform()

        return BgmSpmTransform(np.asarray(patches_info), alphas)
