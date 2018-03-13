# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Parameters
# general
THC_CACHING_ALLOCATOR = 0
SEED = 88
BATCH_SIZE = 64
GENERATED_NUM = 10000
GRADIENT_ESTIMATOR = 'REINFORCE' # 'REINFORCE' or 'RELAX' 
# related to data
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 5000
# pre-training
PRE_EPOCH_GEN = 1
PRE_EPOCH_DIS = 1
PRE_ITER_DIS = 3
# adversarial training
UPDATE_RATE = 0.8
TOTAL_BATCH = 2
G_STEPS = 1
D_STEPS = 4
D_EPOCHS = 2

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True if torch.cuda.is_available() else False
print(opt.cuda)

# Generator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20
n_samples = 2 # controls the precision of the probability distribution at the output of the Generator

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
    	#tqdm(#data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:
    	#tqdm(#data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable 
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss

# New functions/classes

# return probability distribution (in [0,1]) at the output of G
def g_output_prob(prob, BATCH_SIZE, g_sequence_len):
    softmax = nn.Softmax(dim=1)
    theta_prime = softmax(prob)
    theta_prime = torch.sum(theta_prime, dim=0).view((-1,))/(BATCH_SIZE*g_sequence_len)
    return theta_prime

# performs a Gumbel-Softmax reparameterization of the input
def gumbel_softmax(theta_prime, VOCAB_SIZE, cuda=False):
    u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE))))
    if cuda:
        u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE)))).cuda()
    z = torch.log(theta_prime) - u
    return z

# categorical re-sampling exactly as in Backpropagating through the void - Appendix B
def categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda=False):
    v = Variable(torch.rand(VOCAB_SIZE))
    if cuda:
        v = Variable(torch.rand(VOCAB_SIZE)).cuda()
    z_tilde = -torch.log(-torch.log(v)/theta_prime - torch.log(v[b]))
    z_tilde[b] = -torch.log(-torch.log(v[b]))
    return z_tilde


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        target_lstm = target_lstm.cuda()

    # Generate toy data using target lstm
    print('Generating data ...')
    generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)
    
    # Load real data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_GEN):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Dsicriminator ...')
    for epoch in range(PRE_EPOCH_DIS):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(PRE_ITER_DIS):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch [%d], loss: %f' % (epoch, loss))

    # Adversarial Training 
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    if GRADIENT_ESTIMATOR == 'REINFORCE':
        rollout = Rollout(generator, UPDATE_RATE)
        gen_gan_loss = GANLoss()
        if opt.cuda:
            gen_gan_loss = gen_gan_loss.cuda()
    gen_gan_optm = optim.Adam(generator.parameters())
    gen_criterion = nn.NLLLoss(size_average=False)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(G_STEPS):

            if GRADIENT_ESTIMATOR == 'REINFORCE':
                samples = generator.sample(BATCH_SIZE, g_sequence_len)
                # construct the input to the generator, add zeros before samples and delete the last column
                zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
                if samples.is_cuda:
                    zeros = zeros.cuda()
                inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
                targets = Variable(samples.data).contiguous().view((-1,))
                # calculate the reward
                rewards = rollout.get_reward(samples, 16, discriminator)
                rewards = Variable(torch.Tensor(rewards))
                if opt.cuda:
                    rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
                prob = generator.forward(inputs)
                loss = gen_gan_loss(prob, targets, rewards)
                gen_gan_optm.zero_grad()
                loss.backward()
                gen_gan_optm.step()

            if GRADIENT_ESTIMATOR == 'RELAX':
                prob = Variable(torch.zeros((BATCH_SIZE*g_sequence_len, VOCAB_SIZE)))
                if opt.cuda:
                    prob = prob.cuda()
                for n in range(n_samples):
                    samples = generator.sample(BATCH_SIZE, g_sequence_len)
                    # construct the input to the generator, add zeros before samples and delete the last column
                    zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
                    if samples.is_cuda:
                        zeros = zeros.cuda()
                    inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
                    prob += generator.forward(inputs)
                prob /= n_samples
                # 3.a
                theta_prime = g_output_prob(prob, BATCH_SIZE, g_sequence_len)
                # 3.b
                z = gumbel_softmax(theta_prime, VOCAB_SIZE, opt.cuda)
                # 3.c
                value, b = torch.max(z, 0)
                # 3.d
                z_tilde = categorical_re_param(theta_prime, VOCAB_SIZE, b, opt.cuda)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
            print('Batch [%d] True Loss: %f' % (total_batch, loss))
        if GRADIENT_ESTIMATOR == 'REINFORCE':
            rollout.update_params()
        
        for _ in range(D_STEPS):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(D_EPOCHS):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)

if __name__ == '__main__':
    main()
