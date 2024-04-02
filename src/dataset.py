import os
from pathlib import Path

import numpy as np
# from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
        
#         # データディレクトリ内のクラスディレクトリを取得(.DS_Storeを除く)
#         self.classes = [dir_name for dir_name in os.listdir(data_dir) if not dir_name.startswith('.')]
        
#         self.image_paths = []
#         self.labels = []
        
#         # 各クラスディレクトリ内の画像ファイルのパスとラベルを取得
#         for i, class_name in enumerate(self.classes):
#             class_dir = os.path.join(data_dir, class_name)
#             for image_name in os.listdir(class_dir):
#                 self.image_paths.append(os.path.join(class_dir, image_name))
#                 self.labels.append(i)
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         # 指定されたインデックスの画像ファイルのパスとラベルを取得
#         image_path = self.image_paths[idx]
#         label = self.labels[idx]
        
#         # 画像ファイルを読み込み、RGBモードに変換
#         image = Image.open(image_path).convert("RGB")
#         image = np.array(image)
        
#         # 変換関数が指定されている場合は、画像に適用
#         if self.transform:
#             image = self.transform(image=image)['image']
        
#         return image, label


class CustomDataset(Dataset):
    
    def __init__(self, image_path_list, label_list=None, transform=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.label_list:
            label = torch.tensor(self.label_list[idx])
            return image, label
        else:
            return image
