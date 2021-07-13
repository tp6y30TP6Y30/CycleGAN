import torch
import torch.nn as nn

class CycleGANLoss(nn.Module):
	def __init__(self):
		super(CycleGANLoss, self).__init__()
		self.L1Loss = nn.L1Loss(reduction = 'mean')
		self.CLS_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
		self.lambda_ = 10.
		
	def forward(self, mode, img_A, img_B, label_A, label_B, pred_fake_img_A, pred_real_img_A, pred_fake_img_B, pred_real_img_B, A2B2A, B2A2B):
		if mode == 'GAN':
			loss_GAN_A2B = self.CLS_loss(pred_fake_img_A, label_A)
			loss_GAN_B2A = self.CLS_loss(pred_fake_img_B, label_B)
			loss_CC_A = self.L1Loss(A2B2A, img_A)
			loss_CC_B = self.L1Loss(B2A2B, img_B)
			total_GAN_loss = loss_GAN_A2B + loss_GAN_B2A + self.lambda_ * (loss_CC_A + loss_CC_B)
			loss_GAN_A2B_data, loss_GAN_B2A_data, loss_CC_A_data, loss_CC_B_data = loss_GAN_A2B.item(), loss_GAN_B2A.item(), loss_CC_A.item(), loss_CC_B.item()
			return total_GAN_loss, loss_GAN_A2B_data, loss_GAN_B2A_data, loss_CC_A_data, loss_CC_B_data

		elif mode == 'Discr':
			loss_Discr_A = self.CLS_loss(pred_fake_img_A, label_B) + self.CLS_loss(pred_real_img_A, label_A)
			loss_Discr_B = self.CLS_loss(pred_fake_img_B, label_A) + self.CLS_loss(pred_real_img_B, label_B)
			total_Discr_loss = loss_Discr_A + loss_Discr_B
			loss_Discr_A_data, loss_Discr_B_data = loss_Discr_A.item(), loss_Discr_B.item()
			return total_Discr_loss, loss_Discr_A_data, loss_Discr_B_data

		