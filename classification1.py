# encoding=utf8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg19 import VGG19

import os
import pandas as pd
import numpy as np
import cv2

CLASS_NUM = 102
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LABELS_FILE = './data/labels.npy'
TRAIN_CSV = './data/split_train.csv'
VAL_CSV = './data/split_val.csv'
SUBMIT_CSV = './submit.csv'
TRAIN_IMAGES_DIR = './data/split_train_images'
VAL_IMAGES_DIR = './data/split_val_images'
TEST_IMAGES_DIR = './data/test_images'
BATCH_SIZE = 58


def get_filenames_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    filenames = df['filename'].values
    labels = df.iloc[:, 2:].values
    return filenames, labels


def train_generator():
    filenames, labels = get_filenames_labels(TRAIN_CSV)
    while True:
        data_count = len(filenames)
        # 生成一个乱序索引
        shuffle_indexes = np.random.permutation(data_count)
        # TODO: rename filenames
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

        # 获取图片
        for batch_index in range(0, batch_num):
            start = batch_index * BATCH_SIZE
            end = (batch_index + 1) * BATCH_SIZE
            batch_filenames = filenames[start: end]
            result_labels = labels[start: end, :]

            result_images = []
            for filename in batch_filenames:
                img = cv2.imread(os.path.join(TRAIN_IMAGES_DIR, filename))
                result_images.append(img / 255.0)

            result_images = np.array(result_images)
            yield result_images, result_labels
            del result_images


def get_validation_data():
    filenames, labels = get_filenames_labels(VAL_CSV)
    images = []
    for filename in filenames:
        img = cv2.imread(os.path.join(VAL_IMAGES_DIR, filename))
        images.append(img / 255.0)
    return np.array(images), labels


def get_test_data():
    filenames = os.listdir(TEST_IMAGES_DIR)
    images = []
    for filename in filenames:
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, filename))
        images.append(img / 255.0)
    return filenames, np.array(images)


def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def build_vgg():
    base_model = VGG19(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                       weights='imagenet',
                       include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    pred = Dense(CLASS_NUM, activation='softmax')(x)
    return keras.Model(inputs=base_model.input, outputs=pred)


def trainable_model(separator: int, model: keras.Model):
    for layer in model.layers[separator:]:
        layer.trainable = True
    for layer in model.layers[:separator]:
        layer.trainable = False
    print(model.summary())
    print('total layers count: {}'.format(len(model.layers)))
    return model


def train(model: keras.Model, epochs: int):
    steps_per_epoch = np.ceil(len(pd.read_csv(TRAIN_CSV)) / BATCH_SIZE)
    model.fit_generator(train_generator(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=get_validation_data())
    model.save('leon_net.h5', include_optimizer=False)
    print('model saved.')
    return model


def predict(model: keras.Model):
    filenames, test_data = get_test_data()
    result = model.predict(test_data, verbose=1)
    to_csv(filenames, result)


def to_csv(filenames: list, result: np.ndarray):
    labels = np.load(LABELS_FILE)
    top_indexes = np.argsort(result, axis=1)[:, -5:]
    top5_result = []
    for row in range(top_indexes.shape[0]):
        row_data = []
        for col in range(top_indexes.shape[1]):
            row_data.append(labels[top_indexes[row, col]])
        top5_result.append(row_data)

    top5_result = np.array(top5_result)
    result_csv = pd.DataFrame({
        'filename': filenames,
        'label1': top5_result[:, 4],
        'label2': top5_result[:, 3],
        'label3': top5_result[:, 2],
        'label4': top5_result[:, 1],
        'label5': top5_result[:, 0],
    })
    result_csv.set_index('filename', drop=True, inplace=True)
    result_csv.to_csv(SUBMIT_CSV)
    print(SUBMIT_CSV + ' saved.')


if __name__ == '__main__':
    # model = build_vgg()
    model = keras.models.load_model('leon_net.h5', compile=False)

    model = trainable_model(18, model)

    # optimizer = keras.optimizers.Adam(lr=0.0001)
    optimizer = keras.optimizers.SGD(lr=0.000001, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', acc_top5])

    model = train(model, 1)
    predict(model)

