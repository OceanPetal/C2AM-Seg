import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import vit_base_patch16_224

class ResNeSt200e_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'resnest200e',
            pretrained=True,
            features_only=True,
            out_indices=(4,)
        )

    def forward(self, x):
        return self.backbone(x)[0]  # [B,2048,H/32,W/32]

class PSPModule(nn.Module):
    def __init__(self, in_channels, bins=(1, 2, 3, 6)):
        super().__init__()

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_channels, in_channels // len(bins), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(bins)),
                nn.ReLU(inplace=True)
            ) for b in bins
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + in_channels // len(bins) * len(bins),
                in_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for stage in self.stages:
            feats.append(
                F.interpolate(stage(x), size=(h, w),
                              mode='bilinear', align_corners=False)
            )
        return self.bottleneck(torch.cat(feats, dim=1))

class CNN_PSP_Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNeSt200e_Encoder()
        self.psp = PSPModule(2048)

    def forward(self, x):
        return self.psp(self.encoder(x))

class DINOv3_ViTB16_Encoder(nn.Module):
    def __init__(self, weight_path, out_dim=2048):
        super().__init__()

        self.vit = vit_base_patch16_224(
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        state_dict = torch.load(weight_path, map_location='cpu')
        self.vit.load_state_dict(state_dict, strict=False)

        # 🔒 冻结 ViT
        for p in self.vit.parameters():
            p.requires_grad = False

        # ✅ learnable semantic adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(self.vit.embed_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 语义降频（非常重要）
        self.semantic_pool = nn.AvgPool2d(2, 2)  # H/32

    def forward(self, x):
        with torch.no_grad():
            B = x.size(0)

            x = self.vit.patch_embed(x)
            cls = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
            x = x + self.vit.pos_embed
            x = self.vit.pos_drop(x)

            for blk in self.vit.blocks:
                x = blk(x)

            x = x[:, 1:, :]  # remove CLS
            h = w = int(x.shape[1] ** 0.5)
            feat = x.permute(0, 2, 1).reshape(B, -1, h, w)

        feat = self.adapter(feat)
        feat = self.semantic_pool(feat)
        return feat   # [B,2048,H/32,W/32]



class CrossPathFusion(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(
            vit_feat, size=cnn_feat.shape[2:],
            mode='bilinear', align_corners=False
        )

        g = self.gate(cnn_feat)  # structure decides
        return cnn_feat + g * vit_feat



class DualPath_CNN_DINO_Segmentor(nn.Module):
    def __init__(self, nclass, dino_weight_path):
        super().__init__()

        self.cnn_branch = CNN_PSP_Branch()
        self.vit_branch = DINOv3_ViTB16_Encoder(dino_weight_path)

        self.fusion = CrossPathFusion(2048)

        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        feat_cnn = self.cnn_branch(x)
        feat_vit = self.vit_branch(x)

        feat = self.fusion(feat_cnn, feat_vit)
        out = self.head(feat)

        return F.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False
        )
