from timeseries.dataset import DataSettings
from timeseries.dataset import NabDataset
from timeseries.args import Args
from timeseries.layers.LSTMGAN import LSTMGenerator, LSTMDiscriminator
from timeseries.logger import Logger

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils.visualize import Dashboard
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

logger = Logger(__file__)

torch.manual_seed(31)
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # select the device


def main(args):
    logger.info("run analize model")
    writer = SummaryWriter("logs/")

    if torch.cuda.is_available():
        logger.info("run with cuda")

    opt_trn = args.get_option(train=True)

    data_settings = DataSettings(args.opt.data)
    dataset = NabDataset(settings=data_settings)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt_trn["batch_size"],
        shuffle=True,
        num_workers=opt_trn["workers"],
    )

    # input dimension is same as number of feature
    latent_vetor = 100
    in_dim = latent_vetor + 2
    # Create generator and discriminator models
    netD = LSTMDiscriminator(in_dim=3, hidden_dim=256, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=1, hidden_dim=256, device=device).to(
        device
    )

    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn["lr"])
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn["lr"])

    criterion = nn.BCELoss().to(device)

    seq_len = len(dataset)

    # Generate fixed noise to be used for visualization

    # Sample both deltas and noise for visualization
    deltas = (
        dataset.sample_deltas(opt_trn["batch_size"])
        .unsqueeze(2)
        .repeat(1, 2 * seq_len, 1)
        .to(device)
    )

    REAL_LABEL = 1
    FAKE_LABEL = 0
    for epoch in range(opt_trn["epochs"]):
        for i, data in enumerate(dataloader, 0):
            niter = epoch * len(dataloader) + i
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)

            dow = dataset.get_dayofweek(i)
            dows = torch.cat(
                (
                    torch.full((batch_size, seq_len, 1), (dow - 1) % 6, device=device),
                    torch.full((batch_size, seq_len, 1), dow, device=device),
                ),
                dim=1,
            )

            prev = dataset.get_prev(i).to(device).unsqueeze(0)
            real = torch.cat((prev, real), dim=1)
            real = torch.cat((real, deltas, dows), dim=2)
            label = (
                torch.full((batch_size, 2 * seq_len, 1), REAL_LABEL, device=device)
                .type(torch.FloatTensor)
                .to(device)
            )

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            label.fill_(REAL_LABEL)
            output = netD(real).type(torch.FloatTensor).to(device)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake data
            noise = torch.randn(batch_size, 2 * seq_len, latent_vetor, device=device)
            deltas = (
                dataset.sample_deltas(batch_size)
                .unsqueeze(2)
                .repeat(1, 2 * seq_len, 1)
                .to(device)
            )
            noise = torch.cat((noise, deltas, dows), dim=2)

            fake = torch.chunk(netG(noise).detach(), 2, dim=1)[1]
            fake = torch.cat((prev, fake), dim=1)
            label.fill_(FAKE_LABEL)

            output = netD(torch.cat((fake, deltas, dows), dim=2)).to(device)

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Visualize discriminator gradients
            # for name, param in netD.named_parameters():
            #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(REAL_LABEL)
            output = netD(torch.cat((fake, deltas, dows), dim=2))
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # #Visualize generator gradients
            # for name, param in netG.named_parameters():
            #     writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)

            ###########################
            # (3) Suprevised updated of G network
            ###########################
            # Report metrics
            if i == 1:
                real_display = real.cpu()
                fake_display = fake.cpu()

            print(
                f"[{epoch}/{opt_trn['epochs']}][{i}/{len(dataloader)}]"
                f"Loss_D {errD.item()}"
            )
            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    opt_trn["epochs"],
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                ),
                end="\r",
            )

            writer.add_scalar("DiscriminatorLoss", errD.item(), niter)
            writer.add_scalar("GeneratorLoss", errG.item(), niter)
            writer.add_scalar("D_of_X", D_x, niter)
            writer.add_scalar("D_of_G_of_z", D_G_z1, niter)

        print()
        ###########################
        # ( ) End of the epoch
        ###########################
        if epoch % 100 == 0:
            visualize(
                batch_size=opt_trn["batch_size"],
                real_display=dataset.denormalize(real_display),
                fake_display=dataset.denormalize(fake_display),
                block=True,
            )
            # Checkpoint
            torch.save(netG, "netG_epoch_%d.pth" % (epoch))
            torch.save(netD, "netD_epoch_%d.pth" % (epoch))

        # loss_D = errD.item()
        # loss_G = errG.item()
        # loss = loss_D + loss_G
        # score = D_x + D_G_z1 + D_G_z2
        # print(
        #     f"[{epoch + 1:3d}/{opt_trn['epochs']:3d}][{i + 1:3d}/{len(dataloader):3d}] "
        #     f"Loss({loss:.4f}) - D {loss_D:.4f}, G {loss_G:.4f} | "
        #     f"Score({score:.4f}) - D(x) {D_x:.4f}, D(G(z)) {D_G_z1:.4f} / {D_G_z2:.4f} |"
        # )

    # seq_len = dataset_trn.window_length  # sequence length is equal to the window length

    return


if __name__ == "__main__":
    # Argument options
    main(Args())
