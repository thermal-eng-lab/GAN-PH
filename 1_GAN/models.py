# Wasserstein GAN with gradient penalty
# Neural Netowork models

import torch 
import torch.nn as nn

# ----- Generator for WGAN-gp ----- #
class Generator(nn.Module):
    """  
    Input  -> latent vector         ([BATCH_SIZE, latent_size, 4, 4, 4])
              volume fraction       ([BATCH_SIZE, 3, 4, 4, 4])
              specific surface area ([BATCH_SIZE, 3, 4, 4, 4])

    Output -> simulated micro structure  ([BATCH_SIZE, 3, 64, 64, 64])
    """
    def __init__(self, latent_size=100):
        super().__init__()
        self.main= nn.Sequential(

            # Layer 01 (b_size, latent_size+3+3, 4, 4, 4) --> (b_size, 512, 6, 6, 6)
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(latent_size+3+3, 512, 3, 1, 2, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            # Layer 02 (b_size, 512, 6, 6, 6) --> (b_size, 256, 10, 10, 10)
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(512, 256, 3, 1, 2, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # Layer 03 (b_size, 256, 10, 10, 10) --> (b_size, 128, 18, 18, 18)
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(256, 128, 3, 1, 2, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Layer 04 (b_size, 128, 18, 18, 18) --> (b_size, 64, 34, 34, 34)
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(128, 64, 3, 1, 2, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Layer 05 (b_size, 64, 34, 34, 34) --> (b_size, 3, 64, 64, 64)
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(64, 3, 3, 1, 3, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, noise, vf_label, ssa_label):
        x = torch.cat([noise,vf_label, ssa_label],dim=1)
        return self.main(x)

# ----- Critic for WGAN-gp ------ #
class Critic(nn.Module):
    """  
    Input  -> simulated or real image ([BATCH_SIZE, 3, 64, 64, 64])

    Output -> real or fake value      ([BATCH_SIZE])
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            # layer 01 (b_size, 3, 64, 64, 64) --> (b_size, 64, 32, 32, 32)
            nn.utils.spectral_norm(nn.Conv3d(3, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer 02 (b_size, 64, 32, 32, 32) --> (b_size, 128, 16, 16, 16)
            nn.utils.spectral_norm(nn.Conv3d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer 03 (b_size, 128, 16, 16, 16) --> (b_size, 256, 8, 8, 8)
            nn.utils.spectral_norm(nn.Conv3d(128, 256, 4, 2, 1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer 04 (b_size, 256, 8, 8, 8) --> (b_size, 512, 4, 4, 4)
            nn.utils.spectral_norm(nn.Conv3d(256, 512, 4, 2, 1, bias=False)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # layer 05 (b_size, 512, 4, 4, 4) --> (b_size, 1, 1, 1, 1)
            nn.utils.spectral_norm(nn.Conv3d(512, 1, 4, 2, 0, bias=False))
            # nn.Dropout(0.25),
            # nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x).squeeze()

# ------ Estimator of the ssa value ----- #
class Estimator(nn.Module):
    def __init__(self, in_ch, ndf):
        super(Estimator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=ndf, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=ndf, out_channels=ndf*2, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm3d(ndf*2),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(ndf*4),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(ndf*8),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=ndf*8, out_channels=ndf*16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(ndf*16),
            nn.LeakyReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=ndf*16, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size()[0],-1)
        return x.squeeze()
    
def weights_init(weight):
    """ 
    : Function of initializing the weight value

    :paran weight -> the parameters of neural network
    """
    classname = weight.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(weight.weight.data, 0.0, 0.02)
        #torch.nn.init.kaiming_normal_(w.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(weight.weight.data, 1.0, 0.02)
        nn.init.constant_(weight.bias.data, 0)
    return