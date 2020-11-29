from random import shuffle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from scipy.io import loadmat
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import csv
import cv2


class dataset_loader:
    def __init__(self, is_mnist):
        if is_mnist:
            self.class_num = 25
            print('loading sign_MNIST loader')
        else:
            self.class_num = 11
            print('loading kinect_leap loader')

    def load_sign_mnist(self, img_size, isGrayScale):
        Sign_MNIST_train = SignLanguageMNIST(is_train=True, img_size=img_size, is_gray_scale=isGrayScale)
        train_loader = DataLoader(dataset=Sign_MNIST_train, batch_size=8, shuffle=False)
        Sign_MNIST_test = SignLanguageMNIST(is_train=False, img_size=img_size, is_gray_scale=isGrayScale)
        test_loader = DataLoader(dataset=Sign_MNIST_test, batch_size=8, shuffle=False)

        return train_loader, test_loader

    def load_kinect_leap(self, img_size, isGrayScale, train_size=1200):
        all_data = self.load_kinect_leap_dataset()
        shuffle(all_data)

        # set train and test set
        train_data = all_data[0:train_size - 1]
        train_set = KinectLeapDataset(train_data, img_size=img_size, is_gray_scale=isGrayScale)
        train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=False)

        test_data = all_data[train_size:len(all_data) - 1]
        test_set = KinectLeapDataset(test_data, img_size=img_size, is_gray_scale=isGrayScale)
        test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False)

        return train_loader, test_loader

    def load_kinect_leap_dataset(self):
        data_path = './data/kinect_leap_dataset/acquisitions/'
        data = []
        # labels = []

        for i in range(1, 15):
            p_n = 'P' + str(i) + '/'
            # gesture label
            for j in range(1, 11):
                g_n = 'G' + str(j) + '/'
                ges_label = j
                # file name
                for k in range(1, 11):
                    f_n = str(k) + '_rgb.png'
                    img = Image.open(data_path + p_n + g_n + f_n)
                    # print("taking ", data_path + p_n + g_n + f_n)
                    img_arr = np.array(img)
                    data.append((ges_label, img_arr))

        return data


class SignLanguageMNIST(Dataset):
    def __init__(self, is_train, img_size, is_gray_scale=True):
        print('loading sign language MNIST')
        if is_train:
            sign_language_path = './data/archive/sign_mnist_train.csv'
        else:
            sign_language_path = './data/archive/sign_mnist_test.csv'

        self.is_gray_scale = is_gray_scale
        self.data = pd.read_csv(sign_language_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = img_size
        self.width = img_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28, 28).astype('uint8')
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.resize((self.height, self.width))
        #print(img_as_img.size)

        if self.is_gray_scale:
            img_as_img = img_as_img.convert('L')

        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)


class KinectLeapDataset(Dataset):
    def __init__(self, data, img_size, is_gray_scale=False):
        print('loading kinect_leap_dataset')

        # data_path = './data/kinect_leap_dataset/kinect_leap.csv'
        # data_path = './data/kinect_leap_dataset/acquisitions/'
        self.data = data
        self.labels = []
        self.is_gray_scale = is_gray_scale
        for label_tp in self.data:
            self.labels.append(label_tp[0])

        # self.data = pd.read_csv(data_path)
        # self.labels = np.asanyarray(self.data.iloc[:, 0])
        self.height = img_size
        self.width = img_size

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data[index][1])
        img_as_img = Image.fromarray(img_as_np)

        # resize the image to (224, 224)
        img_as_img = img_as_img.resize((self.height, self.width))

        if self.is_gray_scale:
            img_as_img = img_as_img.convert('L')

        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)


'''
loader = dataset_loader(True)
train_loader, test_loader = loader.load_sign_mnist(224, False)
labels = []
for i, (data, tar) in enumerate(train_loader):
    #img = transforms.ToPILImage()(data[0]).convert('RGB')
    #img.show()

    print(data.size())
    for lb in tar:
        if lb not in labels:
            labels.append(lb.item())

    if i >= 40:
        break

labels.sort()
print(labels)
print(len(labels))'''