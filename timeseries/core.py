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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualize import visualize
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
    in_dim = latent_vetor + 1
    # Create generator and discriminator models
    netD = LSTMDiscriminator(in_dim=2, hidden_dim=256, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=1, hidden_dim=256, device=device).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn["lr"])
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn["lr"])
    
    criterion = nn.MSELoss().to(device)
    delta_criterion = nn.MSELoss().to(device)
    
    seq_len = len(dataset)

    #Generate fixed noise to be used for visualization
    fixed_noise = torch.randn(opt_trn["batch_size"], seq_len, latent_vetor, device=device).to(device)

    #Sample both deltas and noise for visualization
    deltas = dataset.sample_deltas(opt_trn["batch_size"]).unsqueeze(2).repeat(1, seq_len, 1).to(device)
    fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

    REAL_LABEL = 1
    FAKE_LABEL = 0
    for epoch in range(opt_trn["epochs"]):
        for i, data in enumerate(dataloader, 0):
            niter = epoch * len(dataloader) + i

            if i == 0:
                real_display = data.cpu()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)

            real = torch.cat((real, deltas), dim=2)
            output = netD(real).type(torch.FloatTensor).to(device)
            label = torch.full((batch_size, seq_len, 1), REAL_LABEL, device=device).type(torch.FloatTensor).to(device)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            #Train with fake data
            noise = torch.randn(batch_size, seq_len, latent_vetor, device=device)
            #Sample a delta for each batch and concatenate to the noise for each timestep
            deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1).to(device)
            noise = torch.cat((noise, deltas), dim=2)

            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            output = netD(torch.cat((fake.detach(), deltas), dim=2)).to(device)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            #Visualize discriminator gradients
            for name, param in netD.named_parameters():
                writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(REAL_LABEL) 
            output = netD(torch.cat((fake, deltas), dim=2))
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            #Visualize generator gradients
            for name, param in netG.named_parameters():
                writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)

            ###########################
            # (3) Suprevised updated of G network
            ###########################
            #Report metrics
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                % (epoch, opt_trn["epochs"], i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='\r')

            writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
            writer.add_scalar('GeneratorLoss', errG.item(), niter)
            writer.add_scalar('D_of_X', D_x, niter) 
            writer.add_scalar('D_of_G_of_z', D_G_z1, niter)

        print()
        ###########################
        # ( ) End of the epoch
        ###########################
        if epoch % 50 == 0:
            fake_display = netG(fixed_noise).cpu()
            visualize(
                batch_size=opt_trn["batch_size"],
                real_display=dataset.denormalize(real_display),
                fake_display=dataset.denormalize(fake_display)
            )
            # Checkpoint
            torch.save(netG, 'netG_epoch_%d.pth' % (epoch))
            torch.save(netD, 'netD_epoch_%d.pth' % (epoch))
                   
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

    dataset_test = NabDataset(data_settings=data_settings)

    print("training option")
    opt_test = Args(args.data, train=False)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt_test.batch_size,
        shuffle=False,
        num_workers=int(opt_test.workers),
    )

    # Setup loss function
    criterion = nn.BCELoss().to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)

    generator = netG  # changing reference variable
    discriminator = netD  # changing reference variable

    def Anomaly_score(x, G_z, Lambda=0.1):
        if x.shape != G_z.shape:
            x = x.reshape(G_z.shape)
            print(x.shape, G_z.shape)
            print(x)
        residual_loss = torch.sum(torch.abs(x - G_z))  # Residual Loss

        # x_feature is a rich intermediate feature representation for real data x
        output, x_feature = discriminator(x.to(device))
        # G_z_feature is a rich intermediate feature representation for fake data G(z)
        output, G_z_feature = discriminator(G_z.to(device))

        discrimination_loss = torch.sum(
            torch.abs(x_feature - G_z_feature)
        )  # Discrimination loss

        total_loss = (1 - Lambda) * residual_loss.to(
            device
        ) + Lambda * discrimination_loss
        return total_loss

    loss_list = []
    # y_list = []
    print(len(dataloader_test))
    for i, (x, y) in enumerate(dataloader_test):
        z = Variable(
            nn.init.normal(
                torch.zeros(
                    opt_test.batch_size,
                    dataset_test.window_length,
                    dataset_test.n_feature,
                ),
                mean=0,
                std=0.1,
            ),
            requires_grad=True,
        )

        # z = x
        z_optimizer = torch.optim.Adam([z], lr=1e-2)

        loss = None
        for j in range(50):  # set your interation range
            gen_fake, _ = generator(z.cuda())
            loss = Anomaly_score(Variable(x).cuda(), gen_fake)
            loss.backward()
            z_optimizer.step()

        loss_list.append(loss)  # Store the loss from the final iteration
        # y_list.append(y) # Store the corresponding anomaly label
        if i % 5 == 4:
            print(f"[{i+1:3d}/{len(dataloader_test)}] BCELoss {loss:4.4f}, y={y}")

    THRESHOLD = (
        11.0  # Anomaly score threshold for an instance to be considered as anomaly
    )

    # TIME_STEPS = dataset.window_length
    test_score_df = pd.DataFrame(index=range(dataset_test.data_len))
    test_score_df["loss"] = [
        loss.item() / dataset_test.window_length for loss in loss_list
    ]
    print(test_score_df.shape)
    test_score_df["y"] = dataset_test.y
    print(test_score_df.shape)
    test_score_df["threshold"] = THRESHOLD
    print(test_score_df.shape)
    test_score_df["anomaly"] = test_score_df.loss > test_score_df.threshold
    print(test_score_df.shape)
    test_score_df["t"] = [torch.mean(x).item() for x in dataset_test.x]
    print(test_score_df.shape)

    plt.plot(test_score_df.index, test_score_df.loss, label="loss")
    plt.plot(test_score_df.index, test_score_df.threshold, label="threshold")
    plt.plot(test_score_df.index, test_score_df.y, label="y")
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()

    anomalies = test_score_df[test_score_df.anomaly == True]

    plt.plot(range(dataset_test.data_len), test_score_df["t"], label="value")

    sns.scatterplot(
        anomalies.index,
        anomalies.t,
        color=sns.color_palette()[3],
        s=52,
        label="anomaly",
    )

    plt.plot(range(len(test_score_df["y"])), test_score_df["y"], label="y")
    plt.xticks(rotation=25)
    plt.legend()

    start_end = []
    state = 0
    for idx in test_score_df.index:
        if state == 0 and test_score_df.loc[idx, "y"] == 1:
            state = 1
            start = idx
        if state == 1 and test_score_df.loc[idx, "y"] == 0:
            state = 0
            end = idx
            start_end.append((start, end))

    for s_e in start_end:
        if sum(test_score_df[s_e[0] : s_e[1] + 1]["anomaly"]) > 0:
            for i in range(s_e[0], s_e[1] + 1):
                test_score_df.loc[i, "anomaly"] = 1

    actual = np.array(test_score_df["y"])
    predicted = np.array([int(a) for a in test_score_df["anomaly"]])

    predicted = np.array(predicted)
    actual = np.array(actual)

    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)

    print("True Positive\t", tp)
    print("True Negative\t", tn)
    print("False Positive\t", fp)
    print("False Negative\t", fn)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    cohen_kappa = cohen_kappa_score(predicted, actual)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
    auc_val = auc(false_positive_rate, true_positive_rate)
    roc_auc_val = roc_auc_score(actual, predicted)

    print("Accuracy\t", accuracy)
    print("Precision\t", precision)
    print("Recall\t", recall)
    print("f-measure\t", fmeasure)
    print("cohen_kappa_score\t", cohen_kappa)
    print("auc\t", auc_val)
    print("roc_auc\t", roc_auc_val)

    plt.show()


if __name__ == "__main__":
    # Argument options
    main(Args())
