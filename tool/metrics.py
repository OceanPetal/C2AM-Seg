import numpy as np
import cv2
from scipy.ndimage import binary_dilation


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(ious)
        return MIoU

    def Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return ious

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Dice_Score(self):
        dice_scores = {}
        for i in range(self.num_class):
            tp = np.diag(self.confusion_matrix)[i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            dice_scores[i] = dice
        mean_dice = np.mean(list(dice_scores.values()))
        return mean_dice, dice_scores

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def _extract_boundary(self, mask, dilation_ratio=0.02):
        """
        提取二值mask的边界并膨胀形成band区域
        """
        h, w = mask.shape
        img_diag = (h ** 2 + w ** 2) ** 0.5
        dilation_radius = max(1, int(round(dilation_ratio * img_diag)))

        # 提取边界
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(boundary, contours, -1, 1, thickness=1)

        # 膨胀边界区域
        dilated = binary_dilation(boundary, iterations=dilation_radius).astype(np.uint8)
        return dilated

    def Boundary_IoU(self, gt_image, pre_image, dilation_ratio=0.02, ignore_index=4):
        """
        计算单张图像的 Mean Boundary IoU，跳过 ignore_index 类
        """
        assert gt_image.shape == pre_image.shape
        biou_per_class = []

        for cls in range(self.num_class):
            if cls == ignore_index:
                continue  # 忽略背景或无效类

            gt_mask = (gt_image == cls).astype(np.uint8)
            pred_mask = (pre_image == cls).astype(np.uint8)

            if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                continue  # 类未出现，跳过

            gt_boundary = self._extract_boundary(gt_mask, dilation_ratio)
            pred_boundary = self._extract_boundary(pred_mask, dilation_ratio)

            intersection = np.logical_and(gt_boundary, pred_boundary).sum()
            union = np.logical_or(gt_boundary, pred_boundary).sum()
            biou = intersection / union if union > 0 else 0
            biou_per_class.append(biou)

        mean_biou = np.mean(biou_per_class) if biou_per_class else 0
        return mean_biou, biou_per_class