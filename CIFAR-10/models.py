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
        x = self.conv2_bn(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        # x = F.sigmoid(x)
        x = x.view(-1, 1).squeeze(1)
        return x