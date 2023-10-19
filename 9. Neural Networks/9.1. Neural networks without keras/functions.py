import numpy as np
from loguru import logger
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split



def read_data_mnist():
    # Carga de datos de keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Poner las imagenes planas con dimensión (28*28,1)
    x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
    # Normalizar las imagenes
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Codificar los labels para el ajuste de la red
    y_train = tf.keras.utils.to_categorical(y_train, 10).reshape(-1, 10, 1)
    y_test = tf.keras.utils.to_categorical(y_test, 10).reshape(-1, 10, 1)
    logger.info("Imágenes descargadas")
    return list(zip(x_train, y_train)),list(zip(x_test, y_test))

def read_data_project():

    images = np.load('images_gray2.npy')
    images = np.reshape(images, (1126, 400, 400))
    labels = np.load('labels_gray2.npy')
    # Split into train test
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Carga de datos de keras
    # Poner las imagenes planas con dimensión (400*400,1)
    x_train = x_train.reshape(x_train.shape[0], 400 * 400, 1)
    x_test = x_test.reshape(x_test.shape[0], 400 * 400, 1)
    # Normalizar las imagenes
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Codificar los labels para el ajuste de la red
    y_train = tf.keras.utils.to_categorical(y_train, 2).reshape(-1, 2, 1)
    y_test = tf.keras.utils.to_categorical(y_test, 2).reshape(-1, 2, 1)
    logger.info("Imágenes descargadas")
    return list(zip(x_train, y_train)),list(zip(x_test, y_test))

def predict_and_plot_random_examples(network,test_data, num_images=3):
    for _ in range(num_images):
        # Seleccionar al azar una imagen por rango 
        random_index = np.random.randint(len(test_data))
        x_example, y_label = test_data[random_index]
        # Predecir con feedforward de la clase network
        predicted_digit = np.argmax(network.feedforward(x_example))
        #Graficar  dígito y  el label
        plt.imshow(x_example.reshape(400, 400), cmap='gray')
        plt.title(f"Predicho: {predicted_digit}")
        plt.show()
        # Enter para seguir
        input("Enter para seguir con la siguiente imágen ... :D ")
        clear_output(wait=True)
