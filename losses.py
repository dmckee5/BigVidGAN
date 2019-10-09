import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss

# xiaodan: Hinge Loss w/ k frames per video
def loss_hinge_dis_k_sum(dis_fake, dis_real):
  #xiaodan: dis_fake, dis_real [B,k,1]
  loss_real_batch = torch.sum(F.relu(1. - dis_real),1) #[B,1]
  loss_fake_batch = torch.sum(F.relu(1. + dis_fake),1) #[B,1]
  loss_real = torch.mean(loss_real_batch)
  loss_fake = torch.mean(loss_fake_batch)
  return loss_real, loss_fake

def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis #xiaodan: changed by xiaodan to accomodate k frames for each video
