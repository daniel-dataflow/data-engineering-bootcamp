import torch

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.nn.functional import relu, softmax
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt


class CNN(Module):
    def __init__(self):
        super().__init__()

        # Conv2d(input channel, output channel, kernel)
        self.conv1 = Conv2d(3, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.pool = MaxPool2d(2, 2)

        self.flatten = Flatten()
        # 32 * 32 -> 28 * 28 -> 14 * 14 -> 10 * 10 -> 5 * 5
        # Linear(input, output)
        self.layer01 = Linear(16 * 5 * 5, 120)
        self.layer02 = Linear(120, 64)
        self.layer03 = Linear(64, 10)

    
    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))

        x = self.flatten(x)
        x = relu(self.layer01(x))
        x = relu(self.layer02(x))
        x = self.layer03(x)
        return x


NAME_LIST = ["airplane", "car", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"]
MODEL = CNN()

ACCURACY = 0


def train_model(train_data=None, test_data=None, batch_size=4, learning_rate=0.001, epochs=5):

    global MODEL, ACCURACY
    learning_rate = learning_rate
    batch_size = batch_size
    epochs = epochs

    if train_data == None:
        train = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
        trainset = DataLoader(train, batch_size, shuffle=True)

    else:
        trainset = train_data
    if test_data == None:
        test = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
        testset = DataLoader(test, batch_size, shuffle=True)
    else:
        testset = test_data


    loss_function = CrossEntropyLoss()

    optimizer = Adam(MODEL.parameters(), lr=learning_rate)

    loss_list = list()

    for epoch in range(epochs):

        correct = 0
        total = 0

        for i, data in enumerate(trainset):
            x_train, y_train = data

            # optimizer 초기화
            optimizer.zero_grad()

            # 순전파
            h = MODEL(x_train)
            loss = loss_function(h, y_train)

            # 역전파
            loss.backward()

            # opt
            optimizer.step()

            loss = loss.item()
            loss_list.append(loss)

            if i % 1000 == 0:
                print(f"epoch: {epoch} ({i:5d}) \t loss : {loss:.3f}")


    test_iter = iter(testset)
    x_test, y_test = next(test_iter)

    predict = MODEL(x_test)
    predict

    _, predict_labels = torch.max(predict ,1)
    print(predict_labels)
    print(y_test)

    correct = 0
    total = 0

    # no_grad : 기울기 계산 X
    with torch.no_grad():
        for data in testset:
            x_test, y_test = data
            h = MODEL(x_test)
            _, predicted = torch.max(h.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    ACCURACY = 100 * correct // total
    print(f"acc : {ACCURACY}")


def save_model(path="cnn.pth"):
    torch.save(MODEL.state_dict(), path)


def load_model(path="cnn.pth"):
    global MODEL
    MODEL.load_state_dict(torch.load(path))
    MODEL.eval()

    return MODEL


def predict_model(model, image):
    global MODEL, NAME_LIST

    MODEL.eval()
    with torch.no_grad():
        output = MODEL(image)
        _, predicted = torch.max(output, 1)
        predict_list = [NAME_LIST[label.item()] for label in predicted]
        
        return predict_list


if __name__ == "__main__":
    # train_model()
    # save_model()

    testset = DataLoader(CIFAR10(root="data", train=False, download=True, transform=ToTensor()), 
                         4, shuffle=True)
    test_iter = iter(testset)
    x_test, y_test = next(test_iter)

    model = load_model()
    result = predict_model(model, x_test)
    print(result)