
import tensorflow as tf

import cv2
import pandas as pd
import torch
import torch.nn as nn
from pytorchcv import model_provider
from torchvision import transforms
from torch.utils.data import Dataset


class SignatureLabelDataset(Dataset):
    def __init__(self, path_csv, transform=None):
        self.df = pd.read_csv(path_csv)
        self.df.columns = ['anchor', 'ref', 'label']
        self.transform = transform

    def __getitem__(self, index):
        anchor_path = self.df.iat[index, 0]
        ref_path = self.df.iat[index, 1]
        label = self.df.iat[index, 2]

        anchor_img = cv2.imread(anchor_path)
        ref_img = cv2.imread(ref_path)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            ref_img = self.transform(ref_img)

        return anchor_img, ref_img, label

    def __len__(self):
        return len(self.df)


class Resnet(nn.Module):
    def __init__(self, model_type: str = "resnet18_wd4", num_features: int = 512): #resnet10, resnet18_wd4, resnet18, resnet50
        super().__init__()

        self._model = getattr(model_provider, model_type)(pretrained=True)
        self._model.output = nn.Linear(self._model.output.in_features,
                                       num_features)

    def transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self._model(x)


class BaseExtractor_rev2():
    def __init__(self, model_type="resnet18_wd4", num_features: int = 512):
        self.model_type = model_type
        self.num_features = num_features
        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")

    def load(self, model_path: str):
        self.model = Resnet(model_type=self.model_type, num_features=self.num_features)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def extract(self, image_np):
        img1_data = self.model.transforms()(image_np[0, :])
        img2_data = self.model.transforms()(image_np[1, :])
        img_data = torch.stack((img1_data, img2_data), 0)
        img_embed = self.model(img_data.to(self.device))
        return img_embed



class BaseExtractor():
    def __init__(self, model_type="metric", batch_size=64):
        self.model_type = model_type
        self.batch_size = batch_size

    def load(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def extract(self, image_np):
        return self.model.predict(image_np, batch_size=self.batch_size)
