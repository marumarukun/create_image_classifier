import torch
from torch import nn
import timm
import pytorch_lightning as pl
from torchmetrics import Accuracy


class MyNet(nn.Module):
    """自作モデルの定義"""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)
    

# class MobileNetV3Trainer(pl.LightningModule):
#     def __init__(self, max_epochs, num_classes, class_weights=None):
#         super(MobileNetV3Trainer, self).__init__()
        
#         # 事前学習済みのMobileNetV3モデルを読み込む
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
        
#         # 最後の全結合層を置き換え、出力ユニット数を2に変更
#         model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)
#         self.model = model
        
#         # 事前学習済みの重みを固定
#         for param in self.model.parameters():
#             param.requires_grad = True
        
#         # # 最後の全結合層の重みを学習可能に設定
#         # for param in self.model.classifier.parameters():
#         #     param.requires_grad = True
        
#         self.max_epochs = max_epochs
#         # self.class_weights = class_weights.to(torch.device('mps'))
    
#     def forward(self, x):
#         return self.model(x)
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
        
#         # # クラスごとの重みを計算
#         # class_weights = self.calculate_class_weights(y)
        
#         # # 重み付き交差エントロピー損失関数を使用
#         # loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, y)
#         loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
        
#         # class_weights = self.calculate_class_weights(y)
#         # loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, y)
#         loss = torch.nn.CrossEntropyLoss()(y_hat, y)
#         self.log('val_loss', loss)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=self.max_epochs,
#             eta_min=1e-6
#         )
#         return [optimizer], [scheduler]
    
