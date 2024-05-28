# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-15 -*-
# -*- Last revision: 2024-05-15 (Vincent Roduit)-*-
# -*- python version : 3.12.3 -*-
# -*- Description: Define the custom models -*-

# import libraries
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, img_size, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = img_size
        self.num_classes = num_classes

        super(CNN, self).__init__()

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
        criterion = nn.CrossEntropyLoss()
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
                    predictions = output.argmax(dim=1)
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
                output = output.argmax(dim=1)
                predictions.append(output.cpu())

        return torch.cat(predictions).numpy().ravel()


class Basic_CNN(CNN):
    def __init__(
        self,
        img_size,
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
        super(Basic_CNN, self).__init__(img_size, num_classes)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.LeakyReLU(0.1)

        self.fc = nn.Linear(32 * (self.image_size // 2) * (self.image_size // 2), self.num_classes)

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
    
class Advanced_CNN(CNN):
    def __init__(
        self,
        img_size,
        num_classes,

    ):
        """
        Constructor

        Layers:
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        - Dropout layer with probability 0.1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 64 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 64 filters, kernel size 3, stride 1, padding 1
        - Dropout layer with probability 0.1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 128 filters, kernel size 3, stride 1, padding 1
        - ReLU activation
        - Max pooling layer with kernel size 2, stride 2
        - Convolutional layer with 128 filters, kernel size 3, stride 1, padding 1
        - Dropout layer with probability 0.1
        - ReLU activation
        - Fully connected layer with 1 output
        """
        super(Advanced_CNN, self).__init__(img_size, num_classes)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.dropout1 = nn.Dropout(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dropout2 = nn.Dropout(0.1)
        self.relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.dropout3 = nn.Dropout(0.1)
        self.relu6 = nn.LeakyReLU(0.1)

        self.fc = nn.Linear(128 * (self.image_size // 32) * (self.image_size // 32), self.num_classes)

    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.dropout1(self.conv2(x))))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.dropout2(self.conv4(x))))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.relu6(self.dropout3(self.conv6(x)))
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class CnnRadius(CNN):
    def __init__(self, img_size, num_classes):
        super().__init__(img_size, num_classes)

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc_output = nn.Linear(512 + 1, self.num_classes)

    def forward(self, x, radius):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        radius = radius.unsqueeze(1)
        x = torch.cat((x, radius), dim=1)
        x = self.fc_output(x)
        return x

    def train_model(
    self, optimizer, scheduler, train_loader, val_loader, num_epochs=10):
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
      criterion = nn.CrossEntropyLoss()
      for epoch in range(num_epochs):
          self.train()
          for input_data, target, radius_info in train_loader:  # Modify the loop to iterate over input_data, target, and radius_info
              input_data, target, radius_info = input_data.to(self.device), target.to(self.device), radius_info.to(self.device)  # Move input_data, target, and radius_info to device
              optimizer.zero_grad()
              output = self(input_data, radius_info)  # Pass input_data and radius_info to the model
              loss = criterion(output, target)
              loss.backward()
              optimizer.step()

          # Validation loop
          self.eval()
          with torch.no_grad():
              total_correct = 0
              test_loss = 0
              for input_data, target, radius_info in val_loader:  # Modify the loop to iterate over input_data, target, and radius_info
                  input_data, target, radius_info = input_data.to(self.device), target.to(self.device), radius_info.to(self.device)  # Move input_data, target, and radius_info to device
                  output = self(input_data, radius_info)  # Pass input_data and radius_info to the model
                  predictions = output.argmax(dim=1)
                  total_correct += (predictions == target).sum().item()
                  test_loss += criterion(output, target).item() * len(input_data)

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
            for input_data, radius_info in test_loader:
                input_data, radius_info = input_data.to(self.device), radius_info.to(self.device)
                output = self(input_data, radius_info)
                output = output.argmax(dim=1)
                predictions.append(output.cpu())

        return torch.cat(predictions).numpy().ravel()