import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, prior_size, num_g_filter, image_channels):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(num_g_filter*8, num_g_filter*4, 4, 2, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(num_g_filter*4)
        self.conv2 = nn.ConvTranspose2d(num_g_filter*4, num_g_filter*2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_g_filter*2)
        self.conv3 = nn.ConvTranspose2d(num_g_filter*2, num_g_filter, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_g_filter)
        self.conv4 = nn.ConvTranspose2d(num_g_filter, image_channels, 4, 2, 1, bias=False)
        self.fc = nn.Linear(prior_size, 4*4*num_g_filter*8)
        self.fc_bn = nn.BatchNorm1d(4*4*num_g_filter*8)
        self.ngf = num_g_filter

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = x.view(-1, self.ngf*8, 4, 4)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_d_filter, image_channels, num_classes):
        super(Discriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(image_channels, num_d_filter, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(num_d_filter, num_d_filter*2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_d_filter*2)
        self.conv3 = nn.Conv2d(num_d_filter*2, num_d_filter*4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_d_filter*4)
        self.conv4 = nn.Conv2d(num_d_filter*4, num_d_filter*8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_d_filter*8)
        self.conv5 = nn.Conv2d(num_d_filter*8, 1, 4, 1, 0, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        # x = self.conv2_bn(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        # x = self.conv3_bn(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        # x = self.conv4_bn(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        # x = F.sigmoid(x)
        x = x.view(-1, 1).squeeze(1)
        return x


class NormalCNN(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(NormalCNN, self).__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(image_channels, 64, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(4*4*512, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = self.pool(x)
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = self.pool(x)
        x = (F.relu(self.conv5(x)))
        x = (F.relu(self.conv6(x)))
        x = self.pool(x)
        x = (F.relu(self.conv7(x)))
        x = (F.relu(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 4*4*512)
        x = (F.relu(self.fc1(x)))
        x = (F.relu(self.fc2(x)))
        x = self.fc3(x)
        # x = F.sigmoid(x)
        x = x.view(-1, 1).squeeze(1)
        return x
