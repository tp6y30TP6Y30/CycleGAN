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

    parser.add_argument('--batchsize', default = 8, type = int,
                        help = "The training batchsize")

    parser.add_argument('--lr', default = 2e-4, type = float,
                        help = "The training learning rate")

    parser.add_argument('--test-num', default = 5, type = int,
                        help = "The num of test samples")

    args = parser.parse_args()
    return args

def train(dataloader_A, dataloader_B, GAN_A2B, GAN_B2A, Discr_A, Discr_B, device, optimizer_GAN, optimizer_Discr, criterion, args, epoch, epochs):
    GAN_A2B.train(), GAN_B2A.train(), Discr_A.train(), Discr_B.train()
    accumulate_total_GAN_loss, accumulate_GAN_loss, accumulate_CC_loss, accumulate_total_Discr_loss = 0, 0, 0, 0
    dataloader_len = min(len(dataloader_A), len(dataloader_B))
    for (img_A, label_A), (img_B, label_B) in tqdm(zip(dataloader_A, dataloader_B), total = dataloader_len, ncols = 80, desc = '[Train] {:d}/{:d}'.format(epoch, epochs)):
        img_A, label_A, img_B, label_B = img_A.to(device), label_A.to(device), img_B.to(device), label_B.to(device)

        optimizer_GAN.zero_grad()
        fake_img_B = GAN_A2B(img_A)
        fake_img_A = GAN_B2A(img_B)

        imgs_B = torch.cat([fake_img_B, img_B], dim = 0)
        imgs_A = torch.cat([fake_img_A, img_A], dim = 0)

        pred_imgs_B = Discr_B(imgs_B)
        pred_imgs_A = Discr_A(imgs_A)

        pred_imgs = torch.cat([pred_imgs_B, pred_imgs_A], dim = 0).squeeze(-1)
        fake_labels = torch.cat([label_B, label_A, label_A, label_B], dim = 0)

        B2A2B = GAN_A2B(GAN_B2A(img_B))
        A2B2A = GAN_B2A(GAN_A2B(img_A))
        fake_imgs = torch.cat([B2A2B, A2B2A], dim = 0)
        imgs = torch.cat([img_B, img_A], dim = 0)

        total_GAN_loss, GAN_loss_data, CC_loss_data = criterion('GAN', pred_imgs = pred_imgs, labels = fake_labels, fake_imgs = fake_imgs, imgs = imgs)
        accumulate_total_GAN_loss += total_GAN_loss.item()
        accumulate_GAN_loss += GAN_loss_data
        accumulate_CC_loss += CC_loss_data
        total_GAN_loss.backward()
        optimizer_GAN.step()

        optimizer_Discr.zero_grad()
        fake_img_B = GAN_A2B(img_A)
        fake_img_A = GAN_B2A(img_B)

        imgs_B = torch.cat([fake_img_B, img_B], dim = 0)
        imgs_A = torch.cat([fake_img_A, img_A], dim = 0)

        pred_imgs_B = Discr_B(imgs_B)
        pred_imgs_A = Discr_A(imgs_A)

        pred_imgs = torch.cat([pred_imgs_B, pred_imgs_A], dim = 0).squeeze(-1)
        labels = torch.cat([label_A, label_B, label_B, label_A], dim = 0)

        total_Discr_loss = criterion('Discr', pred_imgs = pred_imgs, labels = labels, fake_imgs = None, imgs = None)
        accumulate_total_Discr_loss += total_Discr_loss.item()
        total_Discr_loss.backward()
        optimizer_Discr.step()

    print('avg_total_GAN_loss: {:.4f} avg_GAN_loss: {:.4f} avg_CC_loss: {:.4f}'.format(
        accumulate_total_GAN_loss / dataloader_len, accumulate_GAN_loss / dataloader_len, accumulate_CC_loss / dataloader_len))
    print('avg_Discr_loss: {:.4f}'.format(accumulate_total_Discr_loss / dataloader_len))

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
    optimizer_GAN = torch.optim.Adam(list(list(GAN_A2B.parameters()) + list(GAN_B2A.parameters())), lr = args.lr, betas = (0.5, 0.999))
    optimizer_Discr = torch.optim.Adam(list(list(Discr_A.parameters()) + list(Discr_B.parameters())), lr = args.lr, betas = (0.5, 0.999))
    criterion = CycleGANLoss().to(device).float()

    # train
    for epoch in range(1, args.epochs + 1):
        train(train_dataloader_A, train_dataloader_B, GAN_A2B, GAN_B2A, Discr_A, Discr_B, device, optimizer_GAN, optimizer_Discr, criterion, args, epoch, args.epochs)
        test(test_dataloader_A, test_dataloader_B, GAN_A2B, GAN_B2A, device, args, epoch, args.epochs)

if __name__ == '__main__':
    main()
