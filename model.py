import tensorflow_datasets as tfds
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import resize
from tensorflow.keras.preprocessing.image import load_img, array_to_img
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
Model=None
def train():

    dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
    print(metadata)
    class_names = [str(i) for i in range(0, 10)]

    test_dataset, train_dataset = dataset['test'], dataset['train']
    train_len = metadata.splits['train'].num_examples
    test_len = metadata.splits['test'].num_examples


    def normalize(data, labels):
        data = tf.cast(data, dtype=tf.float32)
        data /= 255
        return data, labels


    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)
    plt.figure(figsize=(10, 10))
    i = 0
    for (image, label) in test_dataset.take(25):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(class_names[label])
        i += 1
    plt.show()
    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()
    #Keep in memory use cached data set
    print(train_dataset)

    conv1 = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1))
    maxp1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    conv2 = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation=tf.nn.relu)
    maxp2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    inp = tf.keras.layers.Flatten()
    hidden = tf.keras.layers.Dense(128, activation=tf.nn.relu)
    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    model = tf.keras.Sequential([conv1, maxp1, conv2, maxp2, inp, hidden, output])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.cache().repeat().shuffle(train_len).batch(BATCH_SIZE)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=10,
            steps_per_epoch=math.ceil(train_len/BATCH_SIZE))

    test_loss, test_accuracy = model.evaluate(
        test_dataset, steps=math.ceil(test_len/32))
    print('Accuracy on test dataset:', test_accuracy)
    global Model
    Model=model

def predict():
    global Model
    if Model!=None:
        img = load_img("predict3.jpg")
        img = tf.convert_to_tensor(np.asarray(img))
        img = tf.image.rgb_to_grayscale(img)
        image = tf.image.resize(img, (28, 28))
        image = image.numpy().reshape((28, 28))
        # plt.subplot(5, 5, i+1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        # plt.imshow(image)
        # plt.xlabel("done")
        # plt.show()
        # x=input()
        # i += 1



if __name__ == '__main__':
    predict()


