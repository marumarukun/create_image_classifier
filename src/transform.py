import albumentations as A
from albumentations.pytorch import ToTensorV2


def define_transforms(image_size: int) -> dict:
    
    return {
        'train': define_train_transform(image_size),
        'valid': define_valid_transform(image_size)
    }

def define_train_transform(image_size: int) -> A.Compose:
    
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),  # 左右反転
        A.Rotate(limit=10, p=0.5),  # ランダムに-10度から10度の範囲で回転
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def define_valid_transform(image_size: int) -> A.Compose:
    
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
