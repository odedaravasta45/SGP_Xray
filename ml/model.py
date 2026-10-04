import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

NUM_FINDINGS = 14
NUM_CLASSES_DETECTOR = 15
IMG_SIZE_DET = 512

CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening',
    'Pneumothorax', 'Pulmonary fibrosis'
]


class DenseNet121FPNBackbone(nn.Module):
    def __init__(self, densenet):
        super().__init__()
        self.features = densenet.features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 1024],
            out_channels=256,
        )
        self.out_channels = 256

    def forward(self, x):
        f = self.features
        x = f.conv0(x); x = f.norm0(x); x = f.relu0(x); x = f.pool0(x)
        x = f.denseblock1(x); c1 = x; x = f.transition1(x)
        x = f.denseblock2(x); c2 = x; x = f.transition2(x)
        x = f.denseblock3(x); c3 = x; x = f.transition3(x)
        x = f.denseblock4(x); c4 = f.norm5(x); c4 = torch.relu(c4)
        return self.fpn({'stage1': c1, 'stage2': c2, 'stage3': c3, 'stage4': c4})


def build_detector():
    base = densenet121(weights=None)
    backbone = DenseNet121FPNBackbone(base)
    anchors = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES_DETECTOR,
        rpn_anchor_generator=anchors,
        min_size=IMG_SIZE_DET,
        max_size=IMG_SIZE_DET,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=300,
        rpn_nms_thresh=0.7,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )
    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['stage1', 'stage2', 'stage3', 'stage4'],
        output_size=7,
        sampling_ratio=2,
    )
    return model
