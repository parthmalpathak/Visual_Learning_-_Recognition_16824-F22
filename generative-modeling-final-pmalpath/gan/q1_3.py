import os

import torch
import torch.nn as nn
from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    # pass
    loss = torch.nn.BCEWithLogitsLoss()
    discrim_real_target = torch.ones_like(discrim_real).cuda()
    discrim_fake_target = torch.zeros_like(discrim_fake).cuda()
    loss1 = loss(discrim_real, discrim_real_target)
    loss2 = loss(discrim_fake, discrim_fake_target)
    return (loss1 + loss2)/2



def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    # pass
    loss = torch.nn.BCEWithLogitsLoss()
    discrim_fake_target = torch.ones_like(discrim_fake).cuda()
    loss1 = loss(discrim_fake, discrim_fake_target)
    return loss1


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000, 
    ) #log_period = 1000
