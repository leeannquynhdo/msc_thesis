import torch
import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights

model_tiny = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

class convolution(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x = self.conv1(data)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x
    
class encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = convolution(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, data):
        x = self.conv(data)
        p = self.pool(x)
        return x, p

class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = convolution(out_c + out_c, out_c)
        
    def forward(self, data, skip): # skip connections
        x = self.up(data)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """ Encoding """
        self.en1 = encoder(1, 64)
        self.en2 = encoder(64, 128)
        self.en3 = encoder(128, 256)
        self.en4 = encoder(256, 512)

        
        # """ Bottleneck """
        self.bottle = convolution(512, 1024)
        
        # """ Decoding """
        self.de1 = decoder(1024, 512)
        self.de2 = decoder(512, 256)
        self.de3 = decoder(256, 128)
        self.de4 = decoder(128, 64)
        
        """ Classifier """
        self.last = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    
    def forward(self, data):
        """ Encoding """
        s1, p1 = self.en1(data)
        s2, p2 = self.en2(p1)
        s3, p3 = self.en3(p2)
        s4, p4 = self.en4(p3)
        
        """ Bottleneck """
        b = self.bottle(p4)
        
        """ Decoding """
        d1 = self.de1(b, s4)
        d2 = self.de2(d1, s3)
        d3 = self.de3(d2, s2)
        d4 = self.de4(d3, s1)
        
        """ Classifier """
        outs = self.last(d4)
        
        return torch.sigmoid(outs)
    
class unet_convnext(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoding """
        self.encoder = model_tiny.features # model: convnext tiny weights

        # downsampling
        self.stem = self.encoder[0]  # 192
        self.down1 = self.encoder[2] # 384
        self.down2 = self.encoder[4] # 768
        self.down3 = self.encoder[6] # 1536

        # convnext weights
        self.cn1 = self.encoder[1]
        self.cn2 = self.encoder[3]
        self.cn3 = self.encoder[5]
        self.cn4 = self.encoder[7]
        
        """ Decoding """

        """for convnext tiny"""
        self.decode1 = decoder(768, 384)
        self.decode2 = decoder(384, 192)
        self.decode3 = decoder(192, 96)

        self.last = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )


    def forward(self, data):
        """ Encoding """
        s0 = self.stem(data) 
        cn1 = self.cn1(s0)
        s1 = self.down1(cn1) 
        cn2 = self.cn2(s1)
        s2 = self.down2(cn2)
        cn3 = self.cn3(s2)
        s3 = self.down3(cn3)

        """ Decoding """
        d1 = self.decode1(s3, s2)
        d2 = self.decode2(d1, s1)
        d3 = self.decode3(d2, s0)

        """ Classifier """
        output = self.last(d3)

        return torch.sigmoid(output)
    
    
class HistActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, x):
        return self.sigmoid_derivative(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hist(nn.Module):
    def __init__(self,nBins=10,KSize=(3,3),WSize=(3,3)):
        super().__init__()
        self.nBins = nBins
        self.b = nn.Parameter(torch.randn(nBins)).to(device)
        self.K = nn.Parameter(torch.randn(1,1,*KSize)).to(device) # kernel init
        self.W = nn.Parameter(torch.randn(1,1,*WSize)).to(device) # kernel init
        self.act = HistActivation()
        # reordering to save time in forward()
        self.V = torch.cat([self.W for i in range(nBins)],dim=0)
        self.bias = self.b.view(1,nBins,1,1)

    def forward(self, I):
        IK = nn.functional.conv2d(I.to(device), self.K, None, stride=1, padding=1)
        X = nn.functional.conv2d(self.act(self.bias - IK), self.V, None, padding='same', groups=self.nBins)
        return X

class LON(nn.Module):
    def __init__(self, nKernels, nBins, nOut):
        super().__init__()
        self.convs = nn.ModuleList([Hist(nBins) for i in range(nKernels)])
        self.lin = nn.Linear(nKernels*nBins,nOut)

    def forward(self, X):
        X = [h(X) for h in self.convs]
        X = torch.cat(X,1)
        X = torch.permute(self.lin(torch.permute(X,(0,2,3,1))),(0,3,1,2))
        return X
    

class lon_encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = convolution(in_c, out_c)
        
    def forward(self, data):
        x = self.conv(data)
        return x
    

class LON(nn.Module):
    def __init__(self, nKernels, nBins, nOut):
        super().__init__()
        self.convs = nn.ModuleList([Hist(nBins) for i in range(nKernels)])
        self.lin = nn.Linear(nKernels*nBins,nOut)

    def forward(self, X):
        X = [h(X) for h in self.convs]
        X = torch.cat(X,1)
        X = torch.permute(self.lin(torch.permute(X,(0,2,3,1))),(0,3,1,2))
        return X
    
class LON_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2,2))
        
        """ Encoding """
        self.en1 = LON(2,10,64)
        self.en2 = lon_encoder(64, 128)
        self.en3 = lon_encoder(128, 256)
        self.en4 = lon_encoder(256, 512)

        
        # """ Bottleneck """
        self.bottle = convolution(512, 1024)
        
        # """ Decoding """
        self.de1 = decoder(1024, 512)
        self.de2 = decoder(512, 256)
        self.de3 = decoder(256, 128)
        self.de4 = decoder(128, 64)
        
        """ Classifier """
        self.last = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    
    def forward(self, data):
        """ Encoding """
        s1 = self.en1(data)
        p1 = self.pool(s1)
        s2 = self.en2(p1)
        p2 = self.pool(s2)
        s3 = self.en3(p2)
        p3 = self.pool(s3)
        s4 = self.en4(p3)
        p4 = self.pool(s4)
        
        """ Bottleneck """
        b = self.bottle(p4)
        
        """ Decoding """
        d1 = self.de1(b, s4)
        d2 = self.de2(d1, s3)
        d3 = self.de3(d2, s2)
        d4 = self.de4(d3, s1)
        
        """ Classifier """
        outs = self.last(d4)
        
        return torch.sigmoid(outs)