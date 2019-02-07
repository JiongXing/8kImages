import os
from PIL import Image
import pandas as pd
import tarfile
import numpy as np
import keras
import uuid
import cv2
import shutil


IMAGE_SIZE = 224


# 处理成灰度图，缩小为正方形，短边填白
def new_image(desired_size: int, file_path: str):
    im = Image.open(file_path)

    old_size = im.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new(mode="RGB", size=(desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def batch_images(input_dir: str, output_dir: str):
    files = os.listdir(input_dir)
    for name in files:
        input_path = '{}/{}'.format(input_dir, name)
        output_path = '{}/{}'.format(output_dir, name)
        new_image(IMAGE_SIZE, input_path).save(output_path)
    print('batch_images done.')


def handle_train_csv():
    df = pd.read_csv('./data/train.csv')
    df.drop(df.columns[0], axis=1, inplace=True,)
    df.set_index('filename', drop=True, inplace=True)

    # one-hot
    df = pd.get_dummies(df, columns=['label'], prefix='', prefix_sep='')

    # sort
    columns_name = df.columns.values
    columns_name.sort()
    df.to_csv('./data/one_hot.csv', columns=columns_name)


def save_sorted_class():
    df = pd.read_csv('./data/one_hot.csv')
    labels = df.columns[1:]
    labels = labels.values
    print(labels)
    np.save('./data/labels.npy', labels)


def enhance_image():
    df = pd.read_csv('./data/one_hot.csv')
    origin_images_dir = './data/train_images'

    separator = int(len(df) * 0.2)
    # validation data
    val_df = df[:separator]
    val_df.to_csv('./data/split_val.csv')
    val_images_dir = './data/split_val_images'
    for filename in val_df['filename'].values:
        shutil.copyfile(os.path.join(origin_images_dir, filename),
                        os.path.join(val_images_dir, filename))

    # train data
    df = df[separator:]
    new_df = []

    data_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=20
    )

    train_images_dir = './data/split_train_images'
    total_count = df.shape[0]
    for row_index in range(total_count):
        row_data = df.iloc[row_index]
        # origin data
        new_df.append(row_data)
        shutil.copyfile(os.path.join(origin_images_dir, row_data['filename']),
                        os.path.join(train_images_dir, row_data['filename']))
        # enhanced data
        img = cv2.imread(os.path.join(origin_images_dir, row_data['filename']))
        reshape_img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        for _ in range(9):
            new_img = next(data_gen.flow(reshape_img))
            new_img = np.reshape(new_img, img.shape)
            new_name = str(uuid.uuid1()) + '.jpg'
            cv2.imwrite(os.path.join(train_images_dir, new_name), new_img)
            new_data = row_data.copy()
            new_data['filename'] = new_name
            new_df.append(new_data)
        print('{}/{}'.format(row_index, total_count))
    new_df = pd.DataFrame(new_df)
    new_df.to_csv('./data/split_train.csv')


def get_filenames_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    filenames = df['filename'].values
    labels = df.iloc[:, 2:].values
    return filenames, labels


if __name__ == '__main__':
    # batch_images('./data/train_data', './data/train_images')
    # batch_images('./data/test_data', './data/test_images')

    # handle_train_csv()
    # save_sorted_class()

    # enhance_image()

    # get_filenames_labels('./data/split_val.csv')
    pass
