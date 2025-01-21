import pandas as pd
import numpy as np
from fiftyone import zoo
from torch.utils.hipify.hipify_python import value

def initial_table():
    df = pd.read_csv('/home/lyj/fiftyone/open-images-v7/validation/metadata/labels.csv')
    data = np.zeros((10000, 601))
    new_df = pd.DataFrame(data, columns=df.columns[1:])
    values = 'a'
    new_df.insert(loc=0, column='images_name', value=values)
    new_df.to_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/images_to_labels.csv", index=False)

def make_images_to_labels():
    df = pd.read_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/images_to_labels.csv")
    datasets = zoo.load_zoo_dataset(
        name_or_url="open-images-v7",
        split="validation",
        label_types=["classifications"],
        max_samples=10000,
    )
    datasets.persistent = True

    i = 0
    for sample in datasets:
        df.at[i, 'images_name'] = sample.filename
        if sample.positive_labels is not None:
            for classification in sample.positive_labels.classifications:
                label = classification['label']
                df.at[i, label] = 1
        else:
            df.at[i] = None
        i += 1

    df.to_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/images_to_labels.csv", index=False)

df = pd.read_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/images_to_labels.csv")
train_size, val_size, test_size = df.shape[0] * 0.7, df.shape[0] * 0.1, df.shape[0] * 0.2
train_df, val_df, test_df = (
    df[:int(train_size)], df[int(train_size):int(val_size + train_size)], df[int(val_size + train_size):int(test_size + val_size + train_size)])
train_df.to_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/train.csv", index=False)
val_df.to_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/val.csv", index=False)
test_df.to_csv("/home/lyj/fiftyone/open-images-v7/validation/metadata/test.csv", index=False)




