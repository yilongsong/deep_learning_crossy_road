# model.py
# Apr 28, 2023

'''Defines, trains, saves the ConvNet model'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

import os

import random

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 11 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Training

def main():
    model = ConvNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data
    X = []
    y = []

    for folder in ['downsampled_down', 'downsampled_left', 'downsampled_right', 'downsampled_up', 'downsampled_noop']:
        for file in os.listdir('dataset/'+folder):
            if file == '.DS_Store':
                continue
            x = Image.open('dataset/'+folder+'/'+file)
            transform = transforms.ToTensor()
            x = transform(x)
            X.append(x)
            if folder == 'downsampled_up':
                y.append(0)
            if folder == 'downsampled_down':
                y.append(1)
            if folder == 'downsampled_left':
                y.append(2)
            if folder == 'downsampled_right':
                y.append(3)
            if folder == 'downsampled_noop':
                y.append(4)


    # Shuffle X and y
    indices = list(range(len(X)))
    random.shuffle(indices)

    X_shuffled = [X[i] for i in indices]
    y_shuffled = [y[i] for i in indices]

    X_train = X_shuffled[:int(len(X)*0.7)]
    y_train = y_shuffled[:int(len(X)*0.7)]
    X_test = X_shuffled[int(len(X)*0.7):]
    y_test = y_shuffled[int(len(X)*0.7):]

    X_train = torch.stack(X_train).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)

    # Hyperparameters
    num_epochs = 40
    batch_size = 30
    num_batches = len(X_train) // batch_size

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i in range(num_batches):
            # Get a batch of training data
            batch_X = X_train[i * batch_size : (i + 1) * batch_size]
            batch_y = y_train[i * batch_size : (i + 1) * batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            batch_y = batch_y.long()

            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print epoch loss
        epoch_loss = running_loss / num_batches
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, epoch_loss))

    torch.save(model, 'convnet_trained.pth')

    # Evaluation
    model.eval()  # Set the model to evaluation mode
    X_test = torch.stack(X_test).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)

    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / total
        print("Test Accuracy: {:.2f}%".format(accuracy))


if __name__ == "__main__":
    main()