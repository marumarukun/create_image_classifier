import torch
import timm
import pytorch_lightning as pl
from torchmetrics import Accuracy


class EfficientNetV2Trainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # 事前学習済みのEfficientNetV2モデルをtimmで読み込む
        model = timm.create_model('tf_efficientnetv2_s_in21k',
                                  pretrained=True)


if __name__ == '__main__':
    model = EfficientNetV2Trainer()
    print(model)
