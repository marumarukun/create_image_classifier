import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import MobileNetV3Trainer

if __name__ == '__main__':
    # データの前処理を定義
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),  # 左右反転
        A.Rotate(limit=10, p=0.5),  # ランダムに-10度から10度の範囲で回転
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 学習用データと検証用データのデータセットを作成
    train_dataset = CustomDataset(data_dir='./data/train', transform=train_transform)
    val_dataset = CustomDataset(data_dir='./data/val', transform=val_transform)

    # データローダーを作成
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    def calculate_class_weights(labels):
        # ラベルの頻度を計算
        class_counts = torch.bincount(labels)
        
        # クラスの重みを計算
        total_samples = labels.numel()
        class_weights = total_samples / (class_counts.shape[0] * class_counts)
        
        return class_weights
    
    class_weights = calculate_class_weights(torch.tensor(train_dataset.labels))
    print(f"Class weights: {class_weights}")

    # モデルを初期化
    model = MobileNetV3Trainer(max_epochs=10, 
                               num_classes=len(train_dataset.classes),
                               class_weights=class_weights)

    # ModelCheckpointコールバックを設定
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Trainerを設定
    trainer = Trainer(
        max_epochs=10,
        accelerator='gpu',
        log_every_n_steps=100,
        callbacks=[checkpoint_callback]
    )

    # 学習を実行
    trainer.fit(model, train_loader, val_loader)

    # 最良のモデルを保存
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
