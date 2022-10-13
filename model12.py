import warnings
warnings.filterwarnings("ignore")
import os
import sys
from builtins import print
import math
from collections import deque
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models 
from fuse_util import ChannelAttention, Three_fusion ,Two_fusion,SCPM, deep_fusion
#model9--model12 设置了全新的语义空间转化模块 带语义引导 使用two fusion 和 deep fusion 融合分支特征
#—————————————————————杂项函数———————————————————————————————————
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
#————————————————————————————————————————第一阶段编码—————————————————————————————————————————————————————————————————————————————————
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        resnet1 = models.resnet50(pretrained=True)
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1 = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        self.encoder4 = resnet1.layer4#————————————————————————————————RGB encoder
        #[64, 240, 320][256,120,160][512,60,80][1024,30,40][2048,15,20]resnet50
        
    def forward(self,tensor):
        res0 = self.encoder0(tensor)  #[64, 240, 320]
        res1 = self.encoder1(res0)   #[256, 120, 160]
        res2 = self.encoder2(res1)   #[512, 60, 80]
        res3 = self.encoder3(res2)   #[1024, 30, 40] 
        return res0,res1,res2,res3

class Encoder34(nn.Module):
    def __init__(self):
        super(Encoder34,self).__init__()
        resnet1 = models.resnet34(pretrained=True)
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1 = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        self.encoder4 = resnet1.layer4#————————————————————————————————RGB encoder
        #64,240,320   64,120,160   128,60,80   256,30,40   512,15,20  resnet34
        
    def forward(self,tensor):
        res0 = self.encoder0(tensor)  
        res1 = self.encoder1(res0)   
        res2 = self.encoder2(res1)   
        res3 = self.encoder3(res2)   
        return res0,res1,res2,res3
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class ALL_Decoder(nn.Module):
    def __init__(self):
        super(ALL_Decoder,self).__init__()
        
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(256,128)
        self.fusion_conv1 = ConvBnrelu2d_3(128, 128)
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(128,64) #1/4        
        self.fusion_conv2 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(64,64) #1/2       
        self.fusion_conv3 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(64,64) #1
        self.fusion_conv4 = ConvBnrelu2d_3(64, 64)
        self.fusion_conv5 = nn.Conv2d(64, 9, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv5.weight.data)
        
        self.high1 = ConvBnrelu2d_1(256, 128)
        self.high2 = ConvBnrelu2d_1(256, 64)
        self.high3 = ConvBnrelu2d_1(256, 64)

        self.deconv0=TransConvBnLeakyRelu2d(128,64)
        self.deconv1=TransConvBnLeakyRelu2d(64,64)

        self.conv0=ConvBnrelu2d_3(256, 128)
        self.conv1=ConvBnrelu2d_3(128, 64)
        self.conv2=ConvBnrelu2d_3(128, 64)

        self.conv3=ConvBnrelu2d_3(256, 64)

        self.fuse = ConvBnrelu2d_3(256, 256)

    def forward(self, m1,m2,m3,final):  # 64,240,320   64,120,160   128,60,80   256,30,40   512,15,20  
      
        fusion = self.fuse(final)  #256 30 40
        
        high = fusion  #256 30 40

        fusion = self.fusion_decoder0(fusion)  #128 60 80
        com0=fusion#128 60 80
        high1 = F.interpolate(self.high1(high), scale_factor=2, mode='bilinear',align_corners = True) #128 60 80
        m3=high1*m3+m3
        fusion = fusion + m3 + high1  #128 60 80
        fusion = self.fusion_conv1(fusion) #混合

        #_________________________________________________________________________________
        til0=torch.cat((com0,fusion),dim=1) #128 60 80 cat 128 60 80 = 256 60 80
        til0=self.conv0(til0)       #256 60 80-----128 60 80
        #_________________________________________________________________________________

        fusion1=fusion #128 60 80

        fusion = self.fusion_decoder1(fusion)
        high2 = F.interpolate(self.high2(high), scale_factor=4, mode='bilinear',align_corners = True) #64 120 160
        m2=high2*m2+m2
        fusion = fusion + m2 + high2 #64 120 160
        fusion = self.fusion_conv2(fusion)

        #_________________________________________________________________________________
        com1=self.deconv0(com0)    #128 60 80-----#64 120 160      
        til1=torch.cat((com1,fusion),dim=1) #64 120 160 cat 64 120 160 = 128 120 160
        til1=self.conv1(til1)       #128 120 160-----64 120 160
        #_________________________________________________________________________________


        fusion2=fusion #64 120 160

        fusion = self.fusion_decoder2(fusion)
        high3 = F.interpolate(self.high3(high), scale_factor=8, mode='bilinear',align_corners = True) #64 240 320
        m1=high3*m1+m1
        fusion = fusion + m1 + high3 #64 240 320
        fusion = self.fusion_conv3(fusion)

        #_________________________________________________________________________________
        com2=self.deconv1(com1) #64 120 160 ----#64 240 320
        til2=torch.cat((com2,fusion),dim=1) #64 240 320 cat 64 240 320 = 128 240 320
        til2=self.conv2(til2)       #128 240 320-----64 240 320
        #_________________________________________________________________________________
        til0=F.interpolate(til0, scale_factor=4, mode='bilinear',align_corners = True) #128 60 80 ----128 240 320
        til1=F.interpolate(til1, scale_factor=2, mode='bilinear',align_corners = True) #64 120 160 ----64 240 320
        
        til=torch.cat((til0,til1,til2),dim=1)   #128+64+64 240 320 = 256 230 320
        til=self.conv3(til) #256 240 320---64 240 320
        #_________________________________________________________________________________

        fusion3=fusion #64 240 320
        fusion=fusion+til #64 240 320 + 64 240 320

        fusion = self.fusion_decoder3(fusion) #64 240 320-----#64 480 640
        fusion = self.fusion_conv4(fusion) #64 480 640 -----64 480 640

        fusion4=fusion #64 480 640

        fusion_loss = self.fusion_conv5(fusion) #9 480 640     


        return fusion1,fusion2,fusion3,fusion4,fusion_loss 

class ALL_Decoder_final(nn.Module):
    def __init__(self):
        super(ALL_Decoder_final,self).__init__()
        
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(1024,512)
        self.fusion_conv1 = ConvBnrelu2d_3(512, 512)
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(512,256) #1/4        
        self.fusion_conv2 = ConvBnrelu2d_3(256, 256)
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(256,64) #1/2       
        self.fusion_conv3 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(64,64) #1
        self.fusion_conv4 = ConvBnrelu2d_3(64, 64)
        self.fusion_conv5 = nn.Conv2d(64, 9, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv5.weight.data)
        
        self.high1 = ConvBnrelu2d_1(1024, 512)
        self.high2 = ConvBnrelu2d_1(1024, 256)
        self.high3 = ConvBnrelu2d_1(1024, 64)
        self.high4 = ConvBnrelu2d_1(1024, 64)

        self.fuse = ConvBnrelu2d_3(1024, 1024)

    def forward(self, m1,m2,m3,final):  
      
        fusion = self.fuse(final)  #1024 30 40
        
        high = final  #1024 30 40

        fusion = self.fusion_decoder0(fusion)  #512 60 80
        high1 = F.interpolate(self.high1(high), scale_factor=2, mode='bilinear',align_corners = True) #512 60 80
        fusion = fusion + m3 + high1  #512 60 80
        fusion = self.fusion_conv1(fusion) #混合

        fusion1=fusion #512 60 80

        fusion = self.fusion_decoder1(fusion)
        high2 = F.interpolate(self.high2(high), scale_factor=4, mode='bilinear',align_corners = True)
        fusion = fusion + m2 + high2 #256 120 160
        fusion = self.fusion_conv2(fusion)

        fusion2=fusion #256 120 160

        fusion = self.fusion_decoder2(fusion)
        high3 = F.interpolate(self.high3(high), scale_factor=8, mode='bilinear',align_corners = True)
        fusion = fusion + m1 + high3
        fusion = self.fusion_conv3(fusion)

        fusion3=fusion #64 240 320

        fusion = self.fusion_decoder3(fusion)
        fusion = self.fusion_conv4(fusion) #64 480 640

        fusion4=fusion

        fusion_loss = self.fusion_conv5(fusion) #9 480 640     


        return fusion1,fusion2,fusion3,fusion4,fusion_loss 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Encoder_Block(nn.Module):
    def __init__(self):
        super(Encoder_Block,self).__init__()
        #resnet50  [64,240,320] [256,120,160] [512,60,80] [1024,30,40] [2048,15,20]
        resnet1 = models.resnet50(pretrained=True)
        self.encoder0r = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1r = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2r = resnet1.layer2
        self.encoder3r = resnet1.layer3
        self.encoder4r = resnet1.layer4 #————————————————————————————————RGB encoder

        resnet2 = models.resnet50(pretrained=True)
        self.encoder0t = nn.Sequential(resnet2.conv1, resnet2.bn1, resnet2.relu)
        self.encoder1t = nn.Sequential(resnet2.maxpool, resnet2.layer1)
        self.encoder2t = resnet2.layer2
        self.encoder3t = resnet2.layer3
        self.encoder4t = resnet2.layer4 #————————————————————————————————T encoder

        resnet3 = models.resnet50(pretrained=True)
        resnet3.conv1= nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,bias=False) #
        self.encoder0rt = nn.Sequential(resnet3.conv1, resnet3.bn1, resnet3.relu)
        self.encoder1rt = nn.Sequential(resnet3.maxpool, resnet3.layer1)
        self.encoder2rt = resnet3.layer2
        self.encoder3rt = resnet3.layer3
        self.encoder4rt = resnet3.layer4 #————————————————————————————————RGBT encoder
        #————————————————————————————————————————————————————————————————————————————————————————————
        self.fusion_conv0 = ConvBnrelu2d_3(128, 64)
        self.fusion_conv1 = ConvBnrelu2d_3(128, 64)
        self.fusion_conv2 = ConvBnrelu2d_3(64, 64)

        self.channel_atten0=ChannelAttention(256,4)     #64
        self.channel_atten1=ChannelAttention(256,1)      #256
        #self.channel_atten2=ChannelAttention(1024,2)      #512

        self.final_fusion_conv = ConvBnrelu2d_3(1024, 1024)

        self.two_fusion0=Two_fusion(64,1024)
        self.two_fusion1=Two_fusion(256,1024)
        self.two_fusion2=Two_fusion(512,1024)
        #self.three_fusion3=Three_fusion(1024)
        self.deep_fusion=deep_fusion(1024)

        self.conv=ConvBnrelu2d_3(70, 64)
        
    def forward(self,RGB_tensor,T_tensor,RGBT_tensor,p3):
        p64=self.channel_atten0(p3)
        p256=self.channel_atten1(p3)
        #p512=self.channel_atten2(p3)

        RGBT_tensor=torch.cat((RGBT_tensor,RGB_tensor,T_tensor),dim=1)
        RGBT_tensor=self.conv(RGBT_tensor)

        res0r = self.encoder0r(RGB_tensor)       #[3,480,640]------[64,240,320]
        res0t = self.encoder0t(T_tensor)         #[3,480,640]------[64,240,320]
        res0rt= self.encoder0rt(RGBT_tensor)     #[64,480,640]-----[64,240,320]

        detail_res0r = res0r
        detail_res0t = res0t
        
        #细节向语义补充————————————————————————————############
        # fuse1=torch.cat((res0rt,res0t),dim=1)
        # fuse1=self.fusion_conv0(fuse1)
        # fuse2=torch.cat((res0rt,res0r),dim=1)
        # fuse2=self.fusion_conv1(fuse2)
        # res0rt=fuse1+fuse2
        # res0rt=self.fusion_conv2(res0rt)
        #——————————————————————————————————————————############

        #语义引导细节提取一——————————————————————————############
        res0r_atten=res0r*p64
        res0r=res0r+res0r_atten
        res0t_atten=res0t*p64
        res0t=res0t+res0t_atten
        #——————————————————————————————————————————#############
        res1r = self.encoder1r(res0r)            #[256, 120, 160]
        res1t = self.encoder1t(res0t)            #[256, 120, 160]
        res1rt= self.encoder1rt(res0rt)          #[256, 120, 160]

        detail_res1r = res1r
        detail_res1t = res1t

        #语义引导细节提取二—————————————————————————###############
        res1r_atten=res1r*p256
        res1r=res1r+res1r_atten
        res1t_atten=res1t*p256
        res1t=res1t+res1t_atten       
        #——————————————————————————————————————————##############

        res2r = self.encoder2r(res1r)            #[512, 60, 80]
        res2t = self.encoder2t(res1t)            #[512, 60, 80]
        res2rt= self.encoder2rt(res1rt)          #[512, 60, 80]

        detail_res2r = res2r
        detail_res2t = res2t

        #语义引导细节提取三—————————————————————————###############
        # res2r_atten=res2r*p512
        # res2r=res2r+res2r_atten
        # res2t_atten=res2t*p512
        # res2t=res2t+res2t_atten       
        #——————————————————————————————————————————##############

        res3r = self.encoder3r(res2r)            #[1024, 30, 40] 
        res3t = self.encoder3t(res2t)            #[1024, 30, 40] 
        res3rt= self.encoder3rt(res2rt)          #[1024, 30, 40] 

        #细节特征融合
        m0=self.two_fusion0(detail_res0r,detail_res0t,res3rt)
        m1=self.two_fusion1(detail_res1r,detail_res1t,res3rt)
        m2=self.two_fusion2(detail_res2r,detail_res2t,res3rt)
        #最终语义融合
        #m3=res3rt+res3r+res3t
        m3=self.deep_fusion(res3r,res3t,res3rt)

        return m0,m1,m2,m3
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
class TFNet(nn.Module):
    # resnet50   [64,240,320]   [256,120,160]  [512,60,80]   [1024,30,40]  
    #64,240,320   64,120,160   128,60,80   256,30,40   512,15,20  resnet34
    def __init__(self):
        super(TFNet,self).__init__()

        self.Encoder_RGB = Encoder34()
        self.Encoder_T = Encoder34()
        self.Encoder_Block= Encoder_Block()

        self.Decoder_First=ALL_Decoder()
        self.Decoder_Final=ALL_Decoder_final()

        self.SCPM = SCPM(256)

    def forward(self, input_rgb,input_t):        

        a0,a1,a2,a3=self.Encoder_RGB(input_rgb)             
        b0,b1,b2,b3=self.Encoder_T(input_t)      #64,240,320   64,120,160   128,60,80   256,30,40   512,15,20 

        p0=a0+b0
        p1=a1+b1
        p2=a2+b2
        p3=a3+b3

        p3=self.SCPM(p3)                         #语义增强模块

        fusion0,fusion1,fusion2,fusion3,fusion_loss=self.Decoder_First(p0,p1,p2,p3)   #第一段解码     
        # 512 60 80    256 120 160   64 240 320   64 480 640   9 480 640  
        #____________________________________________________________________________________________
        m0,m1,m2,m3=self.Encoder_Block(input_rgb,input_t,fusion3,p3)
        #____________________________________________________________________________________________
        pre0,pre1,pre2,pre3,pre_loss=self.Decoder_Final(m0,m1,m2,m3)   #最终语义解码

        return pre_loss,fusion_loss
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__=='__main__':
    
    input_rgb=torch.rand(2,3,480,640)
    input_t=torch.rand(2,3,480,640)
    
    text_net=TFNet()
    
    pre_loss,fusion_loss=text_net(input_rgb,input_t)

    print(pre_loss.size(),fusion_loss.size())
    
