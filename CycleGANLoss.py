import torch
import torch.nn as nn

class CycleGANLoss(nn.Module):
	def __init__(self):
		super(CycleGANLoss, self).__init__()
		self.L1Loss = nn.L1Loss(reduction = 'mean')
		self.CLS_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
		self.lambda_ = 10.
		
	def forward(self, mode, pred_imgs, labels, imgs, fake_imgs):
		if mode == 'GAN':
			GAN_loss = self.CLS_loss(pred_imgs, labels)
			CC_loss = self.lambda_ * self.L1Loss(imgs, fake_imgs)
			total_GAN_loss = GAN_loss + CC_loss
			return total_GAN_loss, GAN_loss.item(), CC_loss.item()

		elif mode == 'Discr':
			Discr_loss = self.CLS_loss(pred_imgs, labels)
			return Discr_loss
		