import torch
import torch.utils.data as tud
from torchvision import datasets
from torchvision.transforms import ToTensor


def reshape_mnist_images(images: torch.Tensor) -> torch.Tensor:
    """
    reshape the MNIST images to a flattened version
    :param images: shape (batch_size, 1, 28, 28)
    :return: shape (batch_size, 784, 1)
    """
    return images.reshape(images.size(0), -1, 1)


def read_mnist(datapath="./data", split_fraction=None):
    """
    Read MNIST data
    """
    if split_fraction is None:
        split_fraction = [0.8, 0.2]

    training_data = datasets.MNIST(
        root=datapath,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=datapath,
        train=False,
        download=True,
        transform=ToTensor()
    )

    training_data, validation_data = tud.random_split(training_data, split_fraction)

    return training_data, validation_data, test_data
