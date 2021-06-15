import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.init as init
from torch.autograd import Variable
import datetime
from nab_data import NabDataset
from recurrent_models import LSTMGenerator, LSTMDiscriminator
from data_setting import DataSettings

## Reference https://github.com/mdabashar/TAnoGAN/blob/master/TAnoGAN_Pytorch.ipynb

class ArgsTrn:
    workers=4
    batch_size=32
    epochs=20
    lr=0.0002
    cuda = True
    manualSeed=2

def main():
    opt_trn=ArgsTrn()

    torch.manual_seed(opt_trn.manualSeed)
    cudnn.benchmark = True

    data_settings = DataSettings("data.yml")
    dataset = NabDataset(data_settings=data_settings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_trn.batch_size,
                                             shuffle=True, num_workers=int(opt_trn.workers))

    # check the dataset shape
    print(dataset.x.shape, dataset.y.shape)


    device = torch.device("cuda:0" if opt_trn.cuda else "cpu") # select the device
    seq_len = dataset.window_length # sequence length is equal to the window length
    in_dim = dataset.n_feature # input dimension is same as number of feature
    print(device, seq_len, in_dim)

    # Create generator and discriminator models
    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)

    print("|Discriminator Architecture|\n", netD)
    print("|Generator Architecture|\n", netG)
    
        
    # Setup loss function
    criterion = nn.BCELoss().to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)

    real_label = 1
    fake_label = 0

    for epoch in range(opt_trn.epochs):
        for i, (x,y) in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            #Train with real data
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            output,_ = netD.forward(real)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()
            
            #Train with fake data
            noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
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
            noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
            fake,_ = netG.forward(noise)
            label.fill_(real_label) 
            output,_ = netD.forward(fake)
            errG = criterion(output, label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()
            
            

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
            % (epoch, opt_trn.epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
        print()
if __name__ == "__main__":
    main()