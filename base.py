import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch
import math

class TFNet(nn.Module):
    def __init__(self, n_class):
        super(TFNet, self).__init__()
        model1 = models.resnet50(pretrained = True)
        model2 = models.resnet50(pretrained = True)
############################################################  RGB shallow encoder  ##########################################################################
        self.encoder_T_conv1 = model1.conv1
        self.encoder_T_bn1 = model1.bn1
        self.encoder_T_relu = model1.relu
        self.encoder_T_maxpool = model1.maxpool
        self.encoder_T_layer1 = model1.layer1
        self.encoder_T_layer2 = model1.layer2
        self.encoder_T_layer3 = model1.layer3
############################################################  T shallow encoder  #########################################################################
        self.encoder_RGB_conv1 = model2.conv1
        self.encoder_RGB_bn1 = model2.bn1
        self.encoder_RGB_relu = model2.relu
        self.encoder_RGB_maxpool = model2.maxpool
        self.encoder_RGB_layer1 = model2.layer1
        self.encoder_RGB_layer2 = model2.layer2
        self.encoder_RGB_layer3 = model2.layer3
###########################################################  semantic shared trans #######################################################################
        
        self.fuse = ConvBnrelu2d_3(1024, 1024)
###########################################################  fusion decoder #########################################################################
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(1024,512)
        self.fusion_conv1 = ConvBnrelu2d_3(512, 512)
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(512,256)#1/4        
        self.fusion_conv2 = ConvBnrelu2d_3(256, 256)
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(256,64)#1/2       
        self.fusion_conv3 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(64,64)#1
        self.fusion_conv4 = ConvBnrelu2d_3(64, 64)
        self.fusion_conv5 = nn.Conv2d(64, n_class, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv5.weight.data)
        
        self.high1 = ConvBnrelu2d_1(1024, 512)
        self.high2 = ConvBnrelu2d_1(1024, 256)
        self.high3 = ConvBnrelu2d_1(1024, 64)
        self.high4 = ConvBnrelu2d_1(1024, 64)
        
    def forward(self, RGB,T):
#图片已经存成了4通道的，前三是RGB，最后是T，因此用下面两行将输入表示.数据的结构是【batch_size,channel,width,hight】
        rgb2 = RGB
        t2 = T
###################################  encoder  ######################################              
        rgb2 = self.encoder_RGB_conv1(rgb2)
        rgb2 = self.encoder_RGB_bn1(rgb2)
        rgb2 = self.encoder_RGB_relu(rgb2)
        fusion0 = rgb2
        rgb2 = self.encoder_RGB_maxpool(rgb2)           
        rgb2 = self.encoder_RGB_layer1(rgb2)  
        fusion1 = rgb2
        rgb2 = self.encoder_RGB_layer2(rgb2)
        fusion2 = rgb2
        rgb2 = self.encoder_RGB_layer3(rgb2)
        fusion3 = rgb2
        
        t2 = self.encoder_T_conv1(t2)
        t2 = self.encoder_T_bn1(t2)
        t2 = self.encoder_T_relu(t2)
        fusion4 = t2
        t2 = self.encoder_T_maxpool(t2)
        t2 = self.encoder_T_layer1(t2)
        fusion5 = t2
        t2 = self.encoder_T_layer2(t2)
        fusion6 = t2
        t2 = self.encoder_T_layer3(t2)
        fusion7 = t2
        
        m1 = fusion0 + fusion4
        m2 = fusion1 + fusion5
        m3 = fusion2 + fusion6

        fusion = self.fuse(fusion3 + fusion7)
        
        high = fusion
        fusion = self.fusion_decoder0(fusion)
        high1 = F.interpolate(self.high1(high), scale_factor=2, mode='bilinear',align_corners = True)
                
        fusion = fusion + m3 + high1
        fusion = self.fusion_conv1(fusion)
        fusion = self.fusion_decoder1(fusion)
        high2 = F.interpolate(self.high2(high), scale_factor=4, mode='bilinear',align_corners = True)

        fusion = fusion + m2 + high2
        fusion = self.fusion_conv2(fusion)
        fusion = self.fusion_decoder2(fusion)
        high3 = F.interpolate(self.high3(high), scale_factor=8, mode='bilinear',align_corners = True)

        fusion = fusion + m1 + high3
        fusion = self.fusion_conv3(fusion)
        fusion = self.fusion_decoder3(fusion)
        fusion = self.fusion_conv4(fusion)
        fusion = self.fusion_conv5(fusion)

        return fusion
        
class ConvBnrelu2d_3(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class ConvBnrelu2d_1(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class TransConvBnLeakyRelu2d(nn.Module):
    # deconvolution
    # batch normalization
    # Lrelu
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()      
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)  
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)   
                               
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))  
