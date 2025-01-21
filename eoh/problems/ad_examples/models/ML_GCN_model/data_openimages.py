import os
import torch
import numpy as np
import csv
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

df = pd.read_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/classes.csv", header=None)
object_categories = list(df[1])


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


class Open_Images(Dataset):
    def __init__(self, root, setting, transform=None, target_transform=None):
        self.root = root
        self.path_images = os.path.join(root, 'data')
        self.set = setting
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        #download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'metadata')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + setting + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            print("file_csv is not exist.")

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            setting, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        path = path.zfill(6)
        img = Image.open(os.path.join(self.path_images, path )).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)