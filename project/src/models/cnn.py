# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-15 -*-
# -*- Last revision: 2024-05-15 (Vincent Roduit)-*-
# -*- python version : 3.9.18 -*-
# -*- Description: Define the models -*-

# import libraries
import torch
import torch.nn as nn
import constants
from sklearn.metrics import f1_score 

class Basic_CNN(nn.Module):
    def __init__(
        self,
        image_size,
        num_classes,
    ):
        """
        Constructor.

        Layers:
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Fully connected layer with num_classes outputs
        """
        super(Basic_CNN, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.LeakyReLU(0.1)

        self.fc = nn.Linear(32 * (self.image_size // 2) * (self.image_size // 2), num_classes)

    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(
        self, optimizer, scheduler, train_loader, val_loader, num_epochs=10
    ):
        """
        Train the model.

        Args:
            optimizer (torch.optim): Optimizer used for training.
            scheduler (torch.optim.lr_scheduler): Scheduler used for training.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            num_epochs (int): Number of epochs.
        """
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss for multiclass classification
        for epoch in range(num_epochs):
            self.train()
            for input, target in train_loader:
                input, target = input.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validation loop
            self.eval()
            with torch.no_grad():
                total_correct = 0
                test_loss = 0
                for input, target in val_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    output = self(input)
                    predictions = output.argmax(dim=1)  # Change to argmax for multiclass classification
                    total_correct += (predictions == target).sum().item()
                    test_loss += criterion(output, target).item() * len(input)

                test_loss /= len(val_loader.dataset)
                accuracy = total_correct / len(val_loader.dataset)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}, Validation Accuracy: {accuracy:.4f}"
                )

            scheduler.step(test_loss)

    def predict(self, test_loader):
        """
        Compute predictions on the test set.

        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader.

        Returns:
            predictions (np.ndarray): Predictions on the test set.
        """
        self.eval()
        predictions = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for input in test_loader:
                input = input.to(self.device)
                output = self(input)
                output = output.argmax(dim=1)  # Change to argmax for multiclass classification
                predictions.append(output.cpu())

        return torch.cat(predictions).numpy().ravel()