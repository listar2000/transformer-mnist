from utils import read_mnist, read_fashion_mnist
from tqdm import tqdm
import torch.utils.data as tud
import torch.nn as nn
import torch


def validate_model(val_loader, model, loss_fn, device):
    with torch.no_grad():
        total_samples = 0
        correct_samples = 0
        total_loss = 0

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Perform a forward pass
            outputs = model(images)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_samples / total_samples

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def train_image_transformer(epochs=10, batch_size=64, device='gpu'):
    from model import ImageTransformer
    # Check if GPU is available
    if torch.cuda.is_available() and device == 'gpu':
        print("training will be done with device GPU")
        device = torch.device('cuda')
    else:
        print("training will be done with device CPU")
        device = torch.device('cpu')

    # After reading the data
    training_data, validation_data, _ = read_mnist(datapath="./data")

    # Define the DataLoader
    train_loader = tud.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = tud.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    # test_loader = tud.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Define the model
    dim_model = 512  # The dimensionality of the model, you can adjust this value
    num_class = 10  # The number of classes in MNIST
    model = ImageTransformer(dim_model=dim_model, num_class=num_class)
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        print(f"start training epoch {epoch}")
        tqdm_loop = tqdm(train_loader)
        for _, train_data in enumerate(tqdm_loop):
            images, labels = train_data
            images = images.to(device)
            labels = labels.to(device)

            # Perform a forward pass
            outputs = model(images)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Update tqdm description
            tqdm_loop.set_description(f"Training Loss: {loss.item():.4f}")

            # Perform a backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the weights
            optimizer.step()

        # start validating
        validate_model(val_loader, model, loss_fn, device)

    return model


def train_resnet_fashion_mnist(epochs=10, batch_size=64, device='gpu'):
    from resnet.resnet import ResNet
    # Check if GPU is available
    if torch.cuda.is_available() and device == 'gpu':
        print("training will be done with device GPU")
        device = torch.device('cuda')
    else:
        print("training will be done with device CPU")
        device = torch.device('cpu')

    # After reading the data
    training_data, validation_data, _ = read_mnist(datapath="./data")

    # Define the DataLoader
    train_loader = tud.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = tud.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    # test_loader = tud.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Define the model
    num_class = 10  # The number of classes in MNIST
    model = ResNet(in_channels=1, num_classes=10)
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"start training epoch {epoch}")
        tqdm_loop = tqdm(train_loader)
        for _, train_data in enumerate(tqdm_loop):
            images, labels = train_data
            images = images.to(device)
            labels = labels.to(device)

            # Perform a forward pass
            outputs = model(images)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Update tqdm description
            tqdm_loop.set_description(f"Training Loss: {loss.item():.4f}")

            # Perform a backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the weights
            optimizer.step()

        # start validating
        validate_model(val_loader, model, loss_fn, device)

    return model


if __name__ == "__main__":
    trans_model = train_resnet_fashion_mnist(10)
