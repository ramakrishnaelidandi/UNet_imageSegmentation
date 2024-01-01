import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)
    
class UNET(nn.Module):

    def __init__(self,in_channels,out_channels,features=[64,128,256,512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        #down_net
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
        
        #ups_net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,2,2))
            self.ups.append(DoubleConv(feature*2,feature))

        #bottle_neck
        self.bottle_neck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x  = self.pool(x)

        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for up in range(0,len(self.ups),2):
            x = self.ups[up](x)

            if x.shape != skip_connections[up//2].shape:
                x = TF.resize(x,skip_connections[up//2].shape[2:])
            
            x = torch.cat((x,skip_connections[up//2]),dim=1)
            x = self.ups[up+1](x)

        return self.final_conv(x)

def test():
    x = torch.randn((2,3,323,161))
    model = UNET(3,3)
    pred = model(x)
    print(pred.shape,x.shape)
    assert pred.shape == x.shape

if __name__ == "__main__":
    test()
