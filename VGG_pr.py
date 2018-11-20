import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc1_1 = nn.Linear(8192, 2048)
        self.fc1_2 = nn.Linear(2048, 1024)
        self.fc1_3 = nn.Linear(1024, 31)

        self.fc2_1 = nn.Linear(8192, 2048)
        self.fc2_2 = nn.Linear(2048, 1024)
        self.fc2_3 = nn.Linear(1024, 24)

        self.fc3_1 = nn.Linear(8192, 2048)
        self.fc3_2 = nn.Linear(2048, 1024)
        self.fc3_3 = nn.Linear(1024, 34)

        self.fc4_1 = nn.Linear(8192, 2048)
        self.fc4_2 = nn.Linear(2048, 1024)
        self.fc4_3 = nn.Linear(1024, 34)

        self.fc5_1 = nn.Linear(8192, 2048)
        self.fc5_2 = nn.Linear(2048, 1024)
        self.fc5_3 = nn.Linear(1024, 34)

        self.fc6_1 = nn.Linear(8192, 2048)
        self.fc6_2 = nn.Linear(2048, 1024)
        self.fc6_3 = nn.Linear(1024, 34)

        self.fc7_1 = nn.Linear(8192, 2048)
        self.fc7_2 = nn.Linear(2048, 1024)
        self.fc7_3 = nn.Linear(1024, 34)

        # self.fc1_1 = nn.Linear(2048, 1024)
        # self.fc1_2 = nn.Linear(1024, 512)
        # self.fc1_3 = nn.Linear(512, 31)
        #
        # self.fc2_1 = nn.Linear(2048, 1024)
        # self.fc2_2 = nn.Linear(1024, 512)
        # self.fc2_3 = nn.Linear(512, 24)
        #
        # self.fc3_1 = nn.Linear(2048, 1024)
        # self.fc3_2 = nn.Linear(1024, 512)
        # self.fc3_3 = nn.Linear(512, 34)
        #
        # self.fc4_1 = nn.Linear(2048, 1024)
        # self.fc4_2 = nn.Linear(1024, 512)
        # self.fc4_3 = nn.Linear(512, 34)
        #
        # self.fc5_1 = nn.Linear(2048, 1024)
        # self.fc5_2 = nn.Linear(1024, 512)
        # self.fc5_3 = nn.Linear(512, 34)
        #
        # self.fc6_1 = nn.Linear(2048, 1024)
        # self.fc6_2 = nn.Linear(1024, 512)
        # self.fc6_3 = nn.Linear(512, 34)
        #
        # self.fc7_1 = nn.Linear(2048, 1024)
        # self.fc7_2 = nn.Linear(1024, 512)
        # self.fc7_3 = nn.Linear(512, 34)

    def forward(self, x):
        x1 = self.fc1_1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc1_2(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc1_3(x1)

        x2 = self.fc2_1(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc2_2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc2_3(x2)

        x3 = self.fc3_1(x)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.fc3_2(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.fc3_3(x3)

        x4 = self.fc4_1(x)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = self.fc4_2(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = self.fc4_3(x4)

        x5 = self.fc5_1(x)
        x5 = self.relu(x5)
        x5 = self.dropout(x5)
        x5 = self.fc5_2(x5)
        x5 = self.relu(x5)
        x5 = self.dropout(x5)
        x5 = self.fc5_3(x5)

        x6 = self.fc6_1(x)
        x6 = self.relu(x6)
        x6 = self.dropout(x6)
        x6 = self.fc6_2(x6)
        x6 = self.relu(x6)
        x6 = self.dropout(x6)
        x6 = self.fc6_3(x6)

        x7 = self.fc7_1(x)
        x7 = self.relu(x7)
        x7 = self.dropout(x7)
        x7 = self.fc7_2(x7)
        x7 = self.relu(x7)
        x7 = self.dropout(x7)
        x7 = self.fc7_3(x7)

        return x1, x2, x3, x4, x5, x6, x7

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = Feature()
        self.classifier = Classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x1, x2, x3, x4, x5, x6, x7 = self.classifier(x)
        return x1, x2, x3, x4, x5, x6, x7