import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMarginConfidenceLoss(nn.Module):
    """
    自适应边距置信度损失 (Adaptive Margin Confidence Loss)

    核心思想:
    1. 不仅关注正确类别的概率，还关注正确类别与次优类别之间的margin
    2. 使用自适应的置信度阈值，根据预测的确定性动态调整惩罚
    3. 对困难样本（预测不确定的）给予更大的关注

    优势:
    - 比交叉熵更关注类别间的区分度
    - 自动识别和重点优化困难样本
    - 单一损失函数，计算高效
    - 有助于提升边界区域的分割精度

    参数:
    - num_classes: 类别数量
    - margin: 期望的类别间隔，默认0.3
    - temperature: 温度参数，控制损失的平滑度，默认1.0
    - ignore_index: 忽略的标签索引
    """

    def __init__(self, num_classes, margin=0.3, temperature=1.0, ignore_index=255):
        super(AdaptiveMarginConfidenceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] 模型输出的logits
            targets: [B, H, W] ground truth标签

        Returns:
            loss: 标量损失值
        """
        # 获取softmax概率
        probs = F.softmax(logits / self.temperature, dim=1)  # [B, C, H, W]

        # 创建mask，过滤ignore_index
        valid_mask = (targets != self.ignore_index).float()  # [B, H, W]

        # 获取batch size和spatial dimensions
        B, C, H, W = logits.shape

        # 将targets展平并转换为one-hot编码
        targets_flat = targets.view(B, -1)  # [B, H*W]
        valid_mask_flat = valid_mask.view(B, -1)  # [B, H*W]

        # 为ignore_index创建安全的targets（临时设为0，后续会用mask过滤）
        targets_safe = targets_flat.clone()
        targets_safe[targets_safe == self.ignore_index] = 0

        # 转换为one-hot: [B, H*W, C]
        targets_onehot = F.one_hot(targets_safe, num_classes=C).float()

        # 重塑probs: [B, C, H*W] -> [B, H*W, C]
        probs_flat = probs.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 1. 计算正确类别的概率
        correct_class_prob = (probs_flat * targets_onehot).sum(dim=2)  # [B, H*W]

        # 2. 找出每个像素的次优类别概率（排除正确类别）
        # 将正确类别的概率设为-inf，这样不会被选为最大值
        masked_probs = probs_flat.clone()
        masked_probs = masked_probs - targets_onehot * 1e10
        second_best_prob = masked_probs.max(dim=2)[0]  # [B, H*W]

        # 3. 计算margin损失部分
        # margin_loss鼓励正确类别概率超过次优类别至少margin的距离
        margin_violation = self.margin - (correct_class_prob - second_best_prob)
        margin_loss = F.relu(margin_violation)  # 只惩罚违反margin的情况

        # 4. 计算置信度惩罚部分
        # 对于预测不确定的样本（正确类别概率低）给予额外惩罚
        confidence_penalty = -torch.log(correct_class_prob + 1e-7)

        # 5. 自适应权重：根据预测的确定性动态调整两部分的权重
        # 当模型不确定时(correct_class_prob低)，更关注confidence_penalty
        # 当模型较确定时，更关注margin_loss
        uncertainty = 1.0 - correct_class_prob
        adaptive_weight = torch.sigmoid(5.0 * (uncertainty - 0.5))  # 在0.5附近平滑过渡

        # 6. 组合损失
        pixel_loss = adaptive_weight * confidence_penalty + (1 - adaptive_weight) * margin_loss

        # 7. 应用valid_mask并计算平均损失
        pixel_loss = pixel_loss * valid_mask_flat

        # 计算有效像素的平均损失
        num_valid = valid_mask_flat.sum() + 1e-7
        loss = pixel_loss.sum() / num_valid

        return loss


# 使用示例
if __name__ == "__main__":
    # 创建损失函数
    criterion = AdaptiveMarginConfidenceLoss(
        num_classes=4,
        margin=0.3,  # 可调整：更大的margin要求更强的类别区分
        temperature=1.0,  # 可调整：更大的temperature使损失更平滑
        ignore_index=4
    )

    # 模拟数据
    batch_size, num_classes, height, width = 2, 4, 224, 224
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, 5, (batch_size, height, width))

    # 计算损失
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")

    # 测试梯度反传
    loss.backward()
    print("梯度计算成功！")