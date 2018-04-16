import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import time
import argparse

import models



parser = argparse.ArgumentParser()
parser.add_argument('--inputf', required=True, help='path to dataset')
parser.add_argument('--outf', required=True, help='folder to output images')
parser.add_argument('--batchsize', type=int, default=256, help='input batch size')

opt = parser.parse_args()


# parameters
workers = 4
learning_rate = 0.00005
prior_num = 100
cuda = True
clamp_lower = -0.01
clamp_upper = 0.01
LAMBDA = 10 # Gradient penalty lambda hyperparameter
n_critic = 5 # the number of iterations of the critic per generator iteration

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


num_g_filter = 64
num_d_filter = 64

image_size = 64
image_channel = 3

image_pixels = image_size * image_size * image_channel

log_path = opt.outf + 'log.csv'


# make folders
try:
    os.makedirs(opt.outf)
except:
    pass
try:
    os.makedirs(opt.outf)
except:
    pass



# check cuda
if torch.cuda.is_available() and not cuda:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda')


# data
transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

# train data
train_dataset = dset.CIFAR10(root=opt.inputf, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=int(workers))


# creat models
net_d = models.Discriminator(num_d_filter, image_channel, 1)
net_g = models.Generator(prior_num, num_g_filter, image_channel)


# multi gpus
if torch.cuda.device_count() > 1:
    print('Hell Yeah! You Got %d GPUs!!!!'% torch.cuda.device_count())
    net_d = nn.DataParallel(net_d)
    net_g = nn.DataParallel(net_g)


# set cuda
if cuda:
    net_d.cuda()
    net_g.cuda()


# setup optimizer
optimizer_d = optim.RMSprop(net_d.parameters(), lr=learning_rate)
optimizer_g = optim.RMSprop(net_g.parameters(), lr=learning_rate)


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    # print("real_data: ", real_data.size(), fake_data.size())
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# counting numbers
epoch = 0
real_data = 0
critic = 0


while True:
    epoch += 1
    tStart = time.time()

    for k, (data, label) in enumerate(train_loader, 0):
        ############################
        # train discriminator
        ############################
        # train with real
        net_d.zero_grad()

        # real data labels
        all_one_label = torch.FloatTensor(data.size(0)).fill_(1)

        if cuda:
            data = data.cuda()
            all_one_label = all_one_label.cuda()
        data_v = Variable(data)

        output_d_real = net_d(data_v)

        output_d_real = output_d_real.mean()
        output_d_real.backward(all_one_label)


        # train with fake
        # prior noise
        prior_noise = torch.FloatTensor(data.size(0), prior_num).normal_(0, 1)
        # fake data label
        all_mone_label = torch.FloatTensor(data.size(0)).fill_(-1)

        if cuda:
            prior_noise = prior_noise.cuda()
            all_mone_label = all_mone_label.cuda()

        prior_noise_v = Variable(prior_noise)


        g_out_data = net_g(prior_noise_v)
        fake_data = g_out_data
        fake = Variable(fake_data.data)

        output_d_fake = net_d(g_out_data.detach())

        output_d_fake = output_d_fake.mean()
        output_d_fake.backward(all_mone_label)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(net_d, data_v.data, fake.data, data.size(0))
        gradient_penalty.backward()


        loss_d = output_d_real - output_d_fake + gradient_penalty
        wasserstein_d = output_d_real - output_d_fake

        optimizer_d.step()


        critic += 1

        if critic < n_critic:
            continue
        else:
            critic = 0


        ############################
        # clamp parameters to a cube
        ############################
        # for p in net_d.parameters():
        #     p.data.clamp_(clamp_lower, clamp_upper)


        ############################
        # train generatore
        ############################
        net_g.zero_grad()

        output_d_for_g = net_d(fake_data)
        output_d_for_g.backward(all_one_label)
        optimizer_g.step()



    # save images
    vutils.save_image(data, '%s/real_samples.png' % opt.outf, normalize=True)
    vutils.save_image(fake_data.data, '%s/fake_sample_epoch_%04d.png' % (opt.outf, epoch), normalize=True)


    tEnd = time.time()

    cost_time = tEnd - tStart


    print('------------------------------------------------------------------------')
    print('[Epoch: %d] %f secs' % (epoch, cost_time))
    print('Discriminator Loss: %.5f' % (loss_d))
    print('Wasserstein Loss: %.5f' % (wasserstein_d))
