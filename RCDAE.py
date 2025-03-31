import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, UpSampling2D
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import tensorflow as tf

CHANNEL_1 = 16
CHANNEL_2 = 8
CHANNEL_OUTPUT = 3
EPOCHS = 40
BATCH_SIZE =16

NOISE_FACTOR = 0.2#0.4      #考虑改变噪声因子提高图片质量。

def seed_everything(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(33)

def train(x_train_noisy, x_train):
    # input placeholder
    input_image = Input(shape=(372, 492, 3))
    # encoding layer
    x1 = Conv2D(CHANNEL_1, (3, 3), activation='relu', padding="same")(input_image)
    x = Conv2D(CHANNEL_1, (3, 3), activation='relu', padding="same")(x1)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(CHANNEL_2, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # decoding layer
    x = Conv2D(CHANNEL_2, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(CHANNEL_1, (3, 3),activation='relu', padding='same')(x)
    x = Conv2D(CHANNEL_1, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)+x1
    decoded = Conv2D(CHANNEL_OUTPUT, (3, 3), activation='sigmoid', padding='same')(x)

    # build autoencoder, encoder, decoder
    autoencoder = Model(inputs=input_image, outputs=decoded)
    encoder = Model(inputs=input_image, outputs=encoded)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    # Define a callback to save the best model
    checkpoint = ModelCheckpoint(
        filepath = 'best_model.h5',
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1
    )
    # training
    history_record = autoencoder.fit(x_train_noisy, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, callbacks=[checkpoint], verbose=1)
    return encoder, autoencoder, history_record

def add_noise(x_train, x_test):
    x_train_noisy = x_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)     # limit into [0, 1]
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)   # limit into [0, 1]
    return x_train_noisy, x_test_noisy

def plot_accuray(history_record):
    accuracy = history_record.history["acc"]
    loss = history_record.history["loss"]
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
def show_images(decode_images, x_train):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        ax.imshow(x_train[i].reshape(372, 492, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].reshape(372, 492, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(372, 492, 3))
        img = img_to_array(img)
        img = img / 255.0
        images.append(img)
        filenames.append(filename)
    return np.array(images), filenames

def save_images(images, filenames, folder):
    for i, img in enumerate(images):
        img = img.reshape(372,492,3)
        plt.imsave(os.path.join(folder, filenames[i]), img)

import numpy as np
import math

def psnr(img1, img2):
    # 将图片数据转换为浮点型，避免精度丢失
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # 计算均方误差
    mse = np.mean((img1 - img2) ** 2)
    # 如果均方误差为0，返回正无穷，否则计算PSNR
    if mse == 0:
        return float('inf')
    else:
        max_value = 255
        psnr = 20 * math.log10(max_value / math.sqrt(mse))
        return psnr

if __name__ == '__main__':
    # Load your custom dataset
    x_train_1, train_filenames = load_images_from_folder(r'D:/22/pycharm_project/pythonProject/image_HIT_revise/image_600N_HIT_3_revise/train')
    x_test, test_filenames = load_images_from_folder(r'D:/22/pycharm_project/pythonProject/image_HIT_revise/image_600N_HIT_3_revise/test')
    x_val, val_filenames = load_images_from_folder(r'D:/22/pycharm_project/pythonProject//image_HIT_revise/image_600N_HIT_3_revise/val')

    x_train = np.concatenate([x_train_1, x_test, x_val], axis=0)
    # print(x_train.shape)

    # Step3: reshape data, x_train: (60000, 28, 28, 1), x_test: (10000, 28, 28, 1), one row denotes one sample.
    x_train = x_train.reshape(1800,372, 492, 3)
    # print(x_train.shape)
    # Step4: add noisy
    x_train_noisy, x_test_noisy = add_noise(x_train, x_test)

    # Step5： train
    encoder, autoencoder, history_record = train(x_train_noisy=x_train_noisy, x_train=x_train)

    #Load the best model
    best_model = load_model('best_model.h5')
    # show images
    decoded_train = best_model.predict(x_train_1)
    decoded_test = best_model.predict(x_test)
    decoded_val = best_model.predict(x_val)

    save_images(decoded_train, train_filenames, r'D:/22/pycharm_project/pythonProject/image_HIT_revise/image_600N_HIT_3_revise_re/train')
    save_images(decoded_test, test_filenames, r'D:/22/pycharm_project/pythonProject/image_HIT_revise/image_600N_HIT_3_revise_re/test')
    save_images(decoded_val, val_filenames, r'D:/22/pycharm_project/pythonProject/image_HIT_revise/image_600N_HIT_3_revise_re/val')


    psnr_list = []
    for i in range(len(x_test)):
        psnr_value = psnr(x_test[i], decoded_test[i])
        psnr_list.append(psnr_value)
    psnr_array = np.array(psnr_list)
    train_psnr = np.mean(psnr_array)
    print("Train PSNR:", train_psnr)





