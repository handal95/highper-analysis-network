import argparse

from timeseries.dataset import DataSettings
from timeseries.dataset import NabDataset
from timeseries.args import Args
from timeseries.layers.LSTMGAN import LSTMGenerator, LSTMDiscriminator

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

def main(opt):
    print("Setting data")
    data_settings = DataSettings(opt.data)

    print("Prepare dataset")
    # define dataset object and data loader object for NAB dataset
    dataset_trn = NabDataset(data_settings=data_settings)
    dataset_test = NabDataset(data_settings=data_settings)

    print("training option")
    opt_trn= Args(opt.data, train=True)
    opt_test= Args(opt.data, train=False)

    dataloader = torch.utils.data.DataLoader(
        dataset_trn, batch_size=opt_trn.batch_size,
        shuffle=True, num_workers=int(opt_trn.workers)
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opt_test.batch_size, 
        shuffle=False, num_workers=int(opt_test.workers)
    )

    torch.manual_seed(opt_trn.manualSeed)
    cudnn.benchmark = True
    
    device = torch.device("cuda:0" if opt_trn.cuda else "cpu") # select the device
    seq_len = dataset_trn.window_length # sequence length is equal to the window length
    in_dim = dataset_trn.n_feature # input dimension is same as number of feature

    # Create generator and discriminator models
    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    print("|Discriminator Architecture|\n", netD)

    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)
    print("|Generator Architecture|\n", netG)
    
    # Setup loss function
    criterion = nn.BCELoss().to(device)
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)
    
    real_label = 1
    fake_label = 0

    for epoch in range(opt_trn.epochs):
        for i, (x, y) in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            output, _ = netD.forward(real)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()
            
            # Train with fake data
            noise = Variable(nn.init.normal_(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
            fake,_ = netG.forward(noise)
            output,_ = netD.forward(fake.detach()) # detach causes gradient is no longer being computed or stored to save memeory
            label.fill_(fake_label)
            errD_fake = criterion(output, label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise = Variable(nn.init.normal_(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
            fake,_ = netG.forward(noise)
            label.fill_(real_label) 
            output,_ = netD.forward(fake)
            errG = criterion(output, label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()

        loss_D = errD.item()
        loss_G = errG.item()
        loss = loss_D + loss_G
        score = D_x + D_G_z1 + D_G_z2
        print(
            f"[{epoch + 1:3d}/{opt_trn.epochs:3d}][{i + 1:3d}/{len(dataloader):3d}] "
            f"Loss({loss:.4f}) - D {loss_D:.4f}, G {loss_G:.4f} | "
            f"Score({score:.4f}) - D(x) {D_x:.4f}, D(G(z)) {D_G_z1:.4f} / {D_G_z2:.4f} |" 
        )
        
    generator = netG # changing reference variable 
    discriminator = netD # changing reference variable
    
    def Anomaly_score(x, G_z, Lambda=0.1):
        if x.shape != G_z.shape:
            x = x.reshape(G_z.shape)
            print(x.shape, G_z.shape)
            print(x)
        residual_loss = torch.sum(torch.abs(x-G_z)) # Residual Loss
        
        # x_feature is a rich intermediate feature representation for real data x
        output, x_feature = discriminator(x.to(device)) 
        # G_z_feature is a rich intermediate feature representation for fake data G(z)
        output, G_z_feature = discriminator(G_z.to(device)) 
        
        discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature)) # Discrimination loss
        
        total_loss = (1-Lambda)*residual_loss.to(device) + Lambda*discrimination_loss
        return total_loss


    loss_list = []
    #y_list = []
    print(len(dataloader_test))
    for i, (x, y) in enumerate(dataloader_test):
        z = Variable(
            nn.init.normal(
                torch.zeros(
                    opt_test.batch_size,
                    dataset_test.window_length, 
                    dataset_test.n_feature
                ), mean=0, std=0.1)
            , requires_grad=True
        )

        #z = x
        z_optimizer = torch.optim.Adam([z], lr=1e-2)
        
        loss = None
        for j in range(50): # set your interation range
            gen_fake, _ = generator(z.cuda())
            loss = Anomaly_score(Variable(x).cuda(), gen_fake)
            loss.backward()
            z_optimizer.step()

        loss_list.append(loss) # Store the loss from the final iteration
        #y_list.append(y) # Store the corresponding anomaly label
        if i % 5 == 4:
            print(f'[{i+1:3d}/{len(dataloader_test)}] BCELoss {loss:4.4f}, y={y}')
            
    THRESHOLD = 11.0 # Anomaly score threshold for an instance to be considered as anomaly 

    
    #TIME_STEPS = dataset.window_length
    test_score_df = pd.DataFrame(index=range(dataset_test.data_len))
    test_score_df['loss'] = [loss.item()/dataset_test.window_length for loss in loss_list]
    print(test_score_df.shape)
    test_score_df['y'] = dataset_test.y
    print(test_score_df.shape)
    test_score_df['threshold'] = THRESHOLD
    print(test_score_df.shape)
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    print(test_score_df.shape)
    test_score_df['t'] = [torch.mean(x).item() for x in dataset_test.x]
    print(test_score_df.shape)

    plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    plt.plot(test_score_df.index, test_score_df.y, label='y')
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()
    
    
    anomalies = test_score_df[test_score_df.anomaly == True]

    plt.plot(range(dataset_test.data_len), test_score_df['t'], label='value')

    sns.scatterplot(
        anomalies.index,
        anomalies.t,
        color=sns.color_palette()[3],
        s=52,
        label='anomaly'
    )

    plt.plot(range(len(test_score_df['y'])), test_score_df['y'], label='y')
    plt.xticks(rotation=25)
    plt.legend()
    
    start_end = []
    state = 0
    for idx in test_score_df.index:
        if state==0 and test_score_df.loc[idx, 'y']==1:
            state=1
            start = idx
        if state==1 and test_score_df.loc[idx, 'y']==0:
            state = 0
            end = idx
            start_end.append((start, end))

    for s_e in start_end:
        if sum(test_score_df[s_e[0]:s_e[1]+1]['anomaly'])>0:
            for i in range(s_e[0], s_e[1]+1):
                test_score_df.loc[i, 'anomaly'] = 1
                
    actual = np.array(test_score_df['y'])
    predicted = np.array([int(a) for a in test_score_df['anomaly']])

    predicted = np.array(predicted)
    actual = np.array(actual)
    
    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)

    print('True Positive\t', tp)
    print('True Negative\t', tn)
    print('False Positive\t', fp)
    print('False Negative\t', fn)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    cohen_kappa = cohen_kappa_score(predicted, actual)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
    auc_val = auc(false_positive_rate, true_positive_rate)
    roc_auc_val = roc_auc_score(actual, predicted)

    print('Accuracy\t', accuracy)
    print('Precision\t', precision)
    print('Recall\t', recall)
    print('f-measure\t', fmeasure)
    print('cohen_kappa_score\t', cohen_kappa)
    print('auc\t', auc_val)
    print('roc_auc\t', roc_auc_val)
    
    plt.show()
    
    
if __name__ == "__main__":
    # Argument options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config/data_config.yml", help="data.yml path")
    opt = parser.parse_args()

    main(opt)
