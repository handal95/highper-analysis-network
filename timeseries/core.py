import time
import torch
import torch.nn as nn
import torch.optim as optim

from timeseries.args import Args
from timeseries.logger import Logger
from timeseries.bandgan import BandGan
from timeseries.datasets import DataSettings, Dataset
from timeseries.layers.LSTMGAN import LSTMGenerator, LSTMDiscriminator

from utils.loss import GANLoss
from utils.visualize import Dashboard


logger = Logger(__file__)

torch.manual_seed(31)


def main(args):
    logger.info("run analize model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt_trn = args.get_option(train=True)
    data_settings = DataSettings(args.opt.data, train=True)
    dataset = Dataset(settings=data_settings)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt_trn["batch_size"],
        # shuffle=True,
        num_workers=opt_trn["workers"],
    )

    netD = torch.load("netD_latest.pth")
    netG = torch.load("netG_latest.pth")

    hidden_dim = 256
    in_dim = dataset.n_feature
    # netD = LSTMDiscriminator(in_dim=in_dim, hidden_dim=hidden_dim, device=device).to(
    #     device
    # )
    # netG = LSTMGenerator(
    #     in_dim=in_dim, out_dim=in_dim, hidden_dim=hidden_dim, device=device
    # ).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn["lr"])
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn["lr"] * 0.5)
    batch_size = opt_trn["batch_size"]
    seq_len = dataset.seq_len
    in_dim = dataset.n_feature
    bandGan = BandGan(
        batch_size=batch_size,
        seq_len=seq_len,
        in_dim=in_dim,
        device=device,
        dataloader=dataloader,
        dataset=dataset,
        netD=netD,
        netG=netG,
        optimD=optimizerD,
        optimG=optimizerG,
    )

    # bandGan.train(epochs=opt_trn["epochs"])
    netD = bandGan.netD
    netG = bandGan.netG
    # input dimension is same as number of feature
    # Create generator and discriminator models

    criterion_adv = GANLoss(target_real_label=0.9, target_fake_label=0.1).to(device)
    criterion_l1n = nn.SmoothL1Loss().to(device)
    criterion_l2n = nn.MSELoss().to(device)

    dashboard = Dashboard(dataset.dataset)
    for epoch in range(opt_trn["epochs"]):
        runtime = vistime = 0
        running_loss = {"D": 0, "G": 0, "Dx": 0, "DGz1": 0, "DGz2": 0, "l1": 0, "l2": 0}
        for i, data in enumerate(dataloader, 0):
            start_time = time.time()

            # Prepare dataset
            x = data.to(device)
            batch_size, seq_len = x.size(0), x.size(1)
            shape = (batch_size, seq_len, in_dim)

            netD.zero_grad()
            netG.zero_grad()
            ############################
            # (0) Update D network
            ###########################
            # Train with Real Data x
            optimizerD.zero_grad()

            Dx = netD(x)
            errD_real = criterion_adv(Dx, target_is_real=True)
            errD_real.backward()
            optimizerD.step()

            running_loss["Dx"] += errD_real

            # Train with Fake Data z
            optimizerD.zero_grad()

            z = torch.randn((batch_size, seq_len, in_dim)).to(device)
            Gz = netG(z)
            DGz1 = netD(Gz)

            errD_fake = criterion_adv(DGz1, target_is_real=False)
            errD_fake.backward()
            optimizerD.step()

            errD = errD_fake + errD_real

            running_loss["DGz1"] += DGz1.mean().item()
            running_loss["D"] += errD
            # ############################
            # # (2) Update G network: maximize log(D(G(z)))
            # ###########################
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            z = torch.randn(shape).to(device)
            Gz = netG(z)
            DGz2 = netD(Gz)

            errG_ = criterion_adv(DGz2, target_is_real=False)

            gradients = Gz - x
            gradients_sqr = torch.square(gradients)
            gradients_sqr_sum = torch.sum(gradients_sqr)
            gradients_l2_norm = torch.sqrt(gradients_sqr_sum)
            gradients_penalty = torch.square(1 - gradients_l2_norm)

            errl1 = criterion_l1n(Gz, x) * 200.0
            errl2 = criterion_l2n(Gz, x) * 100.0
            errG = errG_ + errl1 + errl2 + gradients_penalty
            errG.backward()

            optimizerG.step()

            running_loss["G"] += errG_
            running_loss["l1"] += errl1
            running_loss["l2"] += errl2
            running_loss["DGz2"] += DGz2.mean().item()

            end_time = time.time()
            runtime += end_time - start_time

            if opt_trn["batch_size"] < 256:
                dashboard.visualize(
                    dataset.times[i],
                    x.cpu(),
                    netG(torch.randn(shape).to(device)).cpu(),
                )
            vistime += time.time() - end_time

            print(
                f"[{epoch}/{opt_trn['epochs']}][{i}/{len(dataloader)}] "
                f"D  {running_loss['D']/(i + 1):.3f}",
                f"G  {running_loss['G']/(i + 1):.3f}",
                f"l1  {running_loss['l1']/(i + 1):.3f}",
                f"l2  {running_loss['l2']/(i + 1):.3f}",
                f"Dx  {running_loss['Dx']/(i + 1):.3f}",
                f"DGz1  {running_loss['DGz1']/(i + 1):.3f}",
                f"DGz2  {running_loss['DGz2']/(i + 1):.3f} ",
                f"|| {(runtime + vistime):.2f}sec",
                end="\r",
            )
        print()
        ###########################
        # ( ) End of the epoch
        ###########################
        # # Checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save(netG, "netG_vanilla_e%d.pth" % (epoch + 1))
            torch.save(netD, "netD_vanilla_e%d.pth" % (epoch + 1))
            torch.save(netG, "netG_latest.pth")
            torch.save(netD, "netD_latest.pth")
    torch.save(netG, "netG_latest.pth")
    torch.save(netD, "netD_latest.pth")
    return


if __name__ == "__main__":
    # Argument options
    main(Args())
