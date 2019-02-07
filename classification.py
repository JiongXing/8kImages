# encoding=utf8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import pandas as pd
import numpy as np
import cv2
from keras.models import Model
from sortedcontainers import SortedSet

CLASS_NUM = 102
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CSV_FILE = './data/train.csv'
CSV_SUBMIT = './data/submit.csv'
DATA_DIR = './data/train_images'
TEST_DATA_DIR = './data/test_images'
EPOCH = 10
BATCH_SIZE = 32


def split_train_validate(data_frame: pd.DataFrame, val_size: float = 0.25):
    separator = int(len(data_frame) * (1 - val_size))
    train = data_frame[:separator]
    val = data_frame[separator:]
    return (np.array(train['filename']), np.array(train['label']),
            np.array(val['filename']), np.array(val['label']))


def one_hot(labels: np.ndarray, sorted_set: np.ndarray):
    num_row = len(labels)
    num_col = len(sorted_set)
    result = np.zeros((num_row, num_col))
    for row in range(0, num_row):
        label = labels[row]
        for col in range(0, num_col):
            if label == sorted_set[col]:
                result[row, col] = 1
    return result


def train_generator(filenames: np.ndarray, labels: np.ndarray,
                    image_generator: ImageDataGenerator):
    while True:
        data_count = len(filenames)
        # 生成一个乱序索引
        shuffle_indexes = np.random.permutation(data_count)
        filenames = filenames[shuffle_indexes]
        labels = labels[shuffle_indexes]

        batch_num = data_count // BATCH_SIZE
        remainder = data_count % BATCH_SIZE

        # 对末尾不足一个batch的数据补全一个batch
        if remainder > 0:
            number_append = BATCH_SIZE - remainder
            filenames = np.append(filenames, filenames[:number_append])
            labels = np.append(labels, labels[:number_append], axis=0)
            batch_num += 1

        print('--- len ---')
        print('filenames len: {}'.format(len(filenames)))
        print('labels len: {}'.format(len(labels)))

        # 获取图片
        for batch_index in range(0, batch_num):
            start = batch_index * BATCH_SIZE
            end = (batch_index + 1) * BATCH_SIZE
            batch_filenames = filenames[start: end]
            batch_labels = labels[start: end, :]
            result_images = []
            result_labels = []
            for index in range(0, BATCH_SIZE):
                img = cv2.imread(os.path.join(DATA_DIR, batch_filenames[index]))
                result_images.append(img / 255.0)
                result_labels.append(batch_labels[index, :])

                # 数据增强，生成多张
                img = np.reshape(img, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
                for _ in range(3):
                    ex_img = next(image_generator.flow(img))
                    ex_img = np.reshape(ex_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
                    result_images.append(ex_img / 255.0)
                    result_labels.append(batch_labels[index, :])

            yield np.array(result_images), np.array(result_labels)


def get_validation_data(filenames: np.ndarray):
    images = []
    for _, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(DATA_DIR, filename))
        img = img / 255.0
        images.append(img)
    return np.array(images)


def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def build_model():
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3]))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=['accuracy', acc_top5])
    return model


def get_test_data():
    filenames = os.listdir(TEST_DATA_DIR)
    images = []
    for filename in filenames:
        img = cv2.imread(os.path.join(TEST_DATA_DIR, filename))
        img = img / 255.0
        images.append(img)
    return filenames, np.array(images)


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE)
    train_filenames, train_labels, val_filenames, val_labels = split_train_validate(df)
    ss = np.array(SortedSet(df['label']))

    train_labels = one_hot(train_labels, ss)
    val_labels = one_hot(val_labels, ss)
    val_data = get_validation_data(val_filenames)

    image_gen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=20)
    generator = train_generator(train_filenames, train_labels, image_gen)
    model = build_model()
    steps_per_epoch = np.ceil(len(train_filenames) / BATCH_SIZE)
    model.fit_generator(generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCH,
                        validation_data=(val_data, val_labels))

    # 预测
    test_filenames, test_data = get_test_data()
    result = model.predict(test_data, verbose=1)

    print(result.shape)

    top5_idx = np.argsort(result, axis=1)
    top5_idx = top5_idx[:, -5:]
    top5_result = []
    for row_index in range(top5_idx.shape[0]):
        row = []
        for index in top5_idx[row_index, :]:
            row.append(ss[index])
        top5_result.append(row)
    top5_result = np.array(top5_result)

    result_df = pd.DataFrame({
        "filename": test_filenames,
        "label1": top5_result[:, 0],
        "label2": top5_result[:, 1],
        "label3": top5_result[:, 2],
        "label4": top5_result[:, 3],
        "label5": top5_result[:, 4],
    }).to_csv('my_result.csv')

    print('end.')
