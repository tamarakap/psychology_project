import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN

from training.dataset_utils import Rescale
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)


class VGGDataset(Dataset):

    def __init__(self, df, apply_mtcnn = False, align_data = False, rescale_size = (160,160)):
        self.df = df.reset_index()
        self.apply_mtcnn = apply_mtcnn
        self.align_data = align_data
        self.rescale_size = rescale_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, 'image_path']

        if not self.apply_mtcnn:
            image = torchvision.io.read_image(img_name)

        else:
            image = Image.open(img_name)
            try:
                image = mtcnn(image)
                #boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            except:
                print (img_name)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        composed_transforms = transforms.Compose([Rescale(self.rescale_size), normalize])
        image = composed_transforms(image)
        img_class = self.df.loc[idx, 'class']
        sample = {'image': image, 'img_class':img_class}
        return sample


class TestDataset(Dataset):
    def __init__(self, df_path, rescale_size = (160,160)):
        self.df =pd.read_csv(df_path).reset_index()
        self.rescale_size = rescale_size
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        composed_transforms = transforms.Compose([Rescale(self.rescale_size), normalize])
        image1 = composed_transforms(torchvision.io.read_image(self.df.loc[idx, '1']))
        image2 = composed_transforms(torchvision.io.read_image(self.df.loc[idx, '2']))
        sample = {'image1': image1, 'image2': image2, 'label': self.df.loc[idx, 'label']}
        return sample


def get_datasets(df_path, config):
    df=pd.read_csv(df_path)
    train_dataset = VGGDataset(df[df['set']=='train'], apply_mtcnn=config.apply_mtcnn, align_data=config.align_data)
    val_dataset = VGGDataset(df[df['set']=='val'], apply_mtcnn=config.apply_mtcnn, align_data=config.align_data)

    return train_dataset , val_dataset
