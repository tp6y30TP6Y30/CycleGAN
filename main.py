import random
from dataloader import CycleGANData
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.GAN import GANetwork
from model.Discriminator import Discriminator
from CycleGANLoss import CycleGANLoss
import os
from os.path import join
import torchvision.utils as utils
from AdaBelief import AdaBelief

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Setup the training settings.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--query', type = str,
                        help = "The directory that contains training img files")

    parser.add_argument('--mode', type = str,
                        help = "Mode: [train] or [test]?")

    parser.add_argument('--pred-path', type = str, default = './predictions',
                        help = "The generated img directory")

    parser.add_argument('--epochs', default = 200, type = int,
                        help = "The total training epochs")

    parser.add_argument('--batchsize', default = 1, type = int,
                        help = "The training batchsize")

    parser.add_argument('--lr', default = 2e-4, type = float,
                        help = "The training learning rate")

    parser.add_argument('--test-num', default = 5, type = int,
                        help = "The num of test samples")

    args = parser.parse_args()
    return args

def train(dataloader_A, dataloader_B, GAN_A2B, GAN_B2A, Discr_A, Discr_B, device, optimizer_GAN, optimizer_Discr, criterion, args, epoch, epochs):
    GAN_A2B.train(), GAN_B2A.train(), Discr_A.train(), Discr_B.train()
    accumulate_GAN_loss, accumulate_GAN_A2B_loss, accumulate_GAN_B2A_loss, accumulate_CC_A_loss, accumulate_CC_B_loss = 0, 0, 0, 0, 0
    accumulate_Discr_loss, accumulate_Discr_A_loss, accumulate_Discr_B_loss = 0, 0, 0
    dataloader_len = min(len(dataloader_A), len(dataloader_B))
    for (img_A, label_A), (img_B, label_B) in tqdm(zip(dataloader_A, dataloader_B), total = dataloader_len, ncols = 80, desc = '[Train] {:d}/{:d}'.format(epoch, epochs)):
        img_A, label_A, img_B, label_B = img_A.to(device), label_A.to(device), img_B.to(device), label_B.to(device)
        optimizer_GAN.zero_grad()
        fake_img_B = GAN_A2B(img_A)
        fake_img_A = GAN_B2A(img_B)
        pred_fake_img_B = Discr_B(fake_img_B)
        pred_fake_img_A = Discr_A(fake_img_A)
        A2B2A = GAN_B2A(GAN_A2B(img_A))
        B2A2B = GAN_A2B(GAN_B2A(img_B))
        total_GAN_loss, loss_GAN_A2B_data, loss_GAN_B2A_data, loss_CC_A_data, loss_CC_B_data = criterion('GAN', img_A = img_A, img_B = img_B, 
            label_A = label_A, label_B = label_B, pred_fake_img_A = pred_fake_img_A, pred_real_img_A = None, pred_fake_img_B = pred_fake_img_B, pred_real_img_B = None, A2B2A = A2B2A, B2A2B = B2A2B)
        accumulate_GAN_loss += total_GAN_loss.item()
        accumulate_GAN_A2B_loss += loss_GAN_A2B_data
        accumulate_GAN_B2A_loss += loss_GAN_B2A_data
        accumulate_CC_A_loss += loss_CC_A_data
        accumulate_CC_B_loss += loss_CC_B_data
        total_GAN_loss.backward()
        optimizer_GAN.step()

        optimizer_Discr.zero_grad()
        fake_img_B = GAN_A2B(img_A)
        fake_img_A = GAN_B2A(img_B)
        pred_real_img_B = Discr_B(img_B)
        pred_real_img_A = Discr_A(img_A)
        pred_fake_img_B = Discr_B(fake_img_B)
        pred_fake_img_A = Discr_A(fake_img_A)
        total_Discr_loss, loss_Discr_A_data, loss_Discr_B_data = criterion('Discr', img_A = None, img_B = None, label_A = label_A, label_B = label_B, 
            pred_fake_img_A = pred_fake_img_A, pred_real_img_A = pred_real_img_A, pred_fake_img_B = pred_fake_img_B, pred_real_img_B = pred_real_img_B, A2B2A = None, B2A2B = None)
        accumulate_Discr_loss += total_Discr_loss.item()
        accumulate_Discr_A_loss += loss_Discr_A_data
        accumulate_Discr_B_loss += loss_Discr_B_data
        total_Discr_loss.backward()
        optimizer_Discr.step()

    print('avg_GAN_loss: {:.4f} avg_GAN_A2B_loss: {:.4f} avg_GAN_B2A_loss: {:.4f} avg_CC_A_loss: {:.4f} avg_CC_B_loss: {:.4f}'.format(accumulate_GAN_loss / dataloader_len, 
        accumulate_GAN_A2B_loss / dataloader_len, accumulate_GAN_B2A_loss / dataloader_len, accumulate_CC_A_loss / dataloader_len, accumulate_CC_B_loss / dataloader_len))
    print('avg_Discr_loss: {:.4f} avg_Discr_A_loss: {:.4f} avg_Discr_B_loss: {:.4f}'.format(accumulate_Discr_loss / dataloader_len, accumulate_Discr_A_loss / dataloader_len, accumulate_Discr_B_loss / dataloader_len))

def test(dataloader_A, dataloader_B, GAN_A2B, GAN_B2A, device, args, epoch, epochs):
    GAN_A2B.eval(), GAN_B2A.eval()
    with torch.no_grad():
        iter_ = 0
        for (img_A, label_A), (img_B, label_B) in tqdm(zip(dataloader_A, dataloader_B), total = args.test_num, ncols = 80, desc = '[Test] {:d}/{:d}'.format(epoch, epochs)):
            img_A, label_A, img_B, label_B = img_A.to(device), label_A.to(device), img_B.to(device), label_B.to(device)
            fake_img_B = GAN_A2B(img_A)
            fake_img_A = GAN_B2A(img_B)
            save_fake_images(fake_img_A, 'fake_A' + str(iter_) + '.jpg', args.pred_path, args.query, epoch)
            save_fake_images(fake_img_B, 'fake_B' + str(iter_) + '.jpg', args.pred_path, args.query, epoch)
            iter_ += 1
        print()

def save_fake_images(fake_images, filename, pred_path, query, epoch):
    save_path = join(pred_path, query, ('epoch_' + str(epoch)))
    os.makedirs(save_path, exist_ok = True)
    utils.save_image(fake_images.data, join(save_path, filename))

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_dataset_A = CycleGANData(args.query, args.mode, 'A')
    train_dataset_B = CycleGANData(args.query, args.mode, 'B')
    train_dataloader_A = DataLoader(train_dataset_A, batch_size = args.batchsize, shuffle = True, num_workers = 4, pin_memory = True, drop_last = True)
    train_dataloader_B = DataLoader(train_dataset_B, batch_size = args.batchsize, shuffle = True, num_workers = 4, pin_memory = True, drop_last = True)

    test_dataset_A = CycleGANData(args.query, 'test', 'A')
    test_dataset_B = CycleGANData(args.query, 'test', 'B')
    test_dataloader_A = DataLoader(test_dataset_A, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True, drop_last = True)
    test_dataloader_B = DataLoader(test_dataset_B, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True, drop_last = True)

    # build model
    GAN_A2B = GANetwork().to(device).float()
    GAN_B2A = GANetwork().to(device).float()
    Discr_A = Discriminator().to(device).float()
    Discr_B = Discriminator().to(device).float()
    optimizer_GAN = AdaBelief(list(list(GAN_A2B.parameters()) + list(GAN_B2A.parameters())), lr = args.lr, betas = (0.5, 0.999), eps = 1e-12)
    optimizer_Discr = AdaBelief(list(list(Discr_A.parameters()) + list(Discr_B.parameters())), lr = args.lr, betas = (0.5, 0.999), eps = 1e-12)
    criterion = CycleGANLoss().to(device).float()

    # train
    for epoch in range(1, args.epochs + 1):
        train(train_dataloader_A, train_dataloader_B, GAN_A2B, GAN_B2A, Discr_A, Discr_B, device, optimizer_GAN, optimizer_Discr, criterion, args, epoch, args.epochs)
        test(test_dataloader_A, test_dataloader_B, GAN_A2B, GAN_B2A, device, args, epoch, args.epochs)

if __name__ == '__main__':
    main()
