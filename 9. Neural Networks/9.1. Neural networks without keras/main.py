from network import Network
from functions import read_data_mnist, predict_and_plot_random_examples

if __name__ == "__main__":
    # Carga de datos
    train_data, test_data  = read_data_mnist()
    # Crear red
    network = Network([784, 30, 30, 10])
    # Entenar red
    network.SGD(train_data, 30, 10, 0.5, test_data)
    # Función para plotear imágenes al azar
    predict_and_plot_random_examples(network, test_data, 4)
