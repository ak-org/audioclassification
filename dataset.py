import torch 
import torchaudio 
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
from PIL import Image

DESIRED_SAMPLE_RATE = 32000
SEGMENT_SIZE = 5 # secs 
WAVEFORM_SEGMENT = DESIRED_SAMPLE_RATE * SEGMENT_SIZE
ROOT_DIR='/Users/aasheish/kaggle/birdclef-torch'
IMG_SIZE=224

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def waveform_to_img(waveform_file):
    waveform = np.load(waveform_file)
    waveform = torch.from_numpy(waveform)
    spec = torchaudio.transforms.MelSpectrogram(sample_rate = DESIRED_SAMPLE_RATE, 
                                                    n_fft = 2048, 
                                                    hop_length = 512,
                                                    center=True,
                                                    pad_mode="reflect",
                                                    power=2.0,
                                                    norm="slaney",
                                                    f_min=16,
                                                    f_max=16386,
                                                    n_mels = IMG_SIZE)(waveform)

    #spec = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=100)(spec)
    spec = torchaudio.functional.amplitude_to_DB(spec, 
                                                 multiplier=10., #10 for power 20 for amplitude
                                                 amin = 1e-9,
                                                 db_multiplier=torch.log10(torch.max(spec.max(), 
                                                                                     torch.tensor(1e-9))
                                                                          ),
                                                 top_db=80)
    #spec = torch.floor(torch.abs(spec)).to(torch.int)
    #spec = torch.log(spec + 1e-9)
    #print(spec.min(), spec.max())
    img = scale_minmax(spec, 0, 255)
    img = img.to(torch.uint8)
    img = img.permute(1,0)
    img = 255 - img
    return img 

def show(img):
    npimage = img.numpy()
    #npimage = 255 - npimage
    npimage = np.expand_dims(npimage, axis=2)
    print(npimage.shape)
    plt.imshow(npimage, interpolation='nearest')
    plt.show()

def get_labels():
    label_to_idx = {}
    train_metadata_df = pd.read_csv(f'{ROOT_DIR}/birdclef-2023/train_metadata.csv')
    train_metadata_df = train_metadata_df[['primary_label', 'filename']]
    train_metadata_df['filename'] = train_metadata_df['filename'].str.split('/').str[1].str.split('.').str[0]
    #print(train_metadata_df[train_metadata_df['filename'] == 'XC363502'].primary_label.values)
    clef_labels = classess = train_metadata_df['primary_label'].unique()
    n_classes = len(clef_labels)
    #print(clef_labels)
    #print(n_classes)
    for i,label in enumerate(clef_labels, 0):
        label_to_idx[label] = i
    #print(label_to_idx.values())
    return train_metadata_df, label_to_idx

def split_train_test():
    class_count_df = pd.read_csv(f'{ROOT_DIR}/samplecount.csv')
    train_data_npy = []
    val_data_npy = []
    i = 0
    for label in class_count_df.label.values:
        npy_files = np.array(sorted(glob.glob(f'{ROOT_DIR}/train_waveform/{label}/*.npy')))
        cnt = len(npy_files)
        train_cnt = int(math.ceil((0.7 * cnt)))
        val_cnt = cnt - train_cnt 
        for j in range(0, train_cnt):
            train_data_npy.append(npy_files[j])
        i += j
        for k in range(0, val_cnt):
            val_data_npy.append(npy_files[j+k])
        i += k
    return train_data_npy, val_data_npy

def count_classes(train_data_npy, val_data_npy):
    train_classes = []
    val_classes = []
    for item in train_data_npy:
        tr_class = item.split('/')[-2]
        train_classes.append(tr_class)
    for item in val_data_npy:
        va_class = item.split('/')[-2]
        val_classes.append(va_class)
    unique, counts = np.unique(train_classes, return_counts=True)
    unique, counts = np.unique(val_classes, return_counts=True)


class BirdclefDataset(Dataset):
    def __init__(self, train_data_npy, val_data_npy, training=True, transform = None) -> None:
        self.train_data_npy = train_data_npy
        self.val_data_npy = val_data_npy
        self.training = training
        _, self.label_to_idx = get_labels()
        self.one_hot_label = np.eye(len(self.label_to_idx), dtype=np.float16)
        self.transform = transform


    def __len__(self) -> int:
        if self.training == True:
            return len(self.train_data_npy)
        else:
            return len(self.val_data_npy)

    def __getitem__(self, index):
        if self.training == True:
            spec = waveform_to_img(self.train_data_npy[index])
            label_text = self.train_data_npy[index].split('/')[-2]
        else:
            spec = waveform_to_img(self.val_data_npy[index])
            label_text = self.val_data_npy[index].split('/')[-2]
        spec = spec.expand(3, *spec.shape[0:])
        img = Image.fromarray(spec.numpy(), 'RGB')
        if self.transform:
            img = self.transform(img)
        label = self.one_hot_label[self.label_to_idx[label_text]]
        return img, label
       
def load_datasets(batch_size = 32):
    train_data_npy, val_data_npy = split_train_test()
    #count_classes(train_data_npy, val_data_npy)
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor()
    ])
    train_data = BirdclefDataset(train_data_npy, val_data_npy, 
                                 training=True, 
                                 transform=data_transform)
    test_data = BirdclefDataset(train_data_npy, val_data_npy, 
                                training=False, 
                                transform=data_transform)
    train_dataloader = DataLoader(train_data, 
                                  batch_size = batch_size, 
                                  num_workers = 4, 
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, 
                                 batch_size = batch_size, 
                                 num_workers = 2, 
                                 shuffle=False)
    #train_features, train_labels = next(iter(train_dataloader))
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")
    #test_features, test_labels = next(iter(test_dataloader))
    #print(f"Feature batch shape: {test_features.size()}")
    #print(f"Labels batch shape: {test_labels.size()}")
    return train_dataloader, test_dataloader

if __name__ == "__main__":

    #spec = waveform_to_img(f'{ROOT_DIR}/train_waveform/abethr1/XC363502_5.npy')
    #show(spec)
    spec = waveform_to_img(f'{ROOT_DIR}/train_waveform/abethr1/XC616997_6.npy')
    show(spec)
    #_, label_dict = get_labels()
    #print(len(label_dict))
    #print(np.eye(len(label_dict)))
    #train_dataloader, test_dataloader = load_datasets()
