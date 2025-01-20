# CNN network model
# Surrogate model for estimating the specific surface area

# -------- import library -------- #
import torch
import torch.nn as nn


# -------- CNN network model -------- #
class Net(nn.Module):
    def __init__(self, in_ch, ndf):
        super(Net, self).__init__()

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
    classname = weight.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(weight.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(weight.weight.data, 1.0, 0.02)
        nn.init.constant_(weight.bias.data, 0)

    return