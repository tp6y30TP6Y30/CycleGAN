import torch
import torch.nn as nn

class CycleGANLoss(nn.Module):
	def __init__(self):
		super(CycleGANLoss, self).__init__()
		self.L1Loss = nn.L1Loss(reduction = 'mean')
		self.MSELoss = nn.MSELoss(reduction = 'mean')
		self.CLS_loss = nn.CrossEntropyLoss(reduction = 'mean')
		self.lambda_ = 10.
		
	def forward(self, mode, pred_imgs, labels, fake_imgs, imgs):
		if mode == 'GAN':
			GAN_loss = self.CLS_loss(pred_imgs, labels)
			CC_loss = self.lambda_ * self.L1Loss(fake_imgs, imgs)
			total_GAN_loss = GAN_loss + CC_loss
			return total_GAN_loss, GAN_loss.item(), CC_loss.item()

		elif mode == 'Discr':
			Discr_loss = self.CLS_loss(pred_imgs, labels)
			return Discr_loss
		