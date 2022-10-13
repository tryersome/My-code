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

#——————————————————————————————————————————————————————————————————————————————————————————————————————
class AFuse(nn.Module):
    def __init__(self,in_channel):
        super(AFuse, self).__init__()
        self.conv1 = nn.Conv2d(3*in_channel, in_channel, 3, padding=1, bias=False)

    def forward(self, rgb_tensor,t_tensor,rgbt_tensor):
        tensor1=rgb_tensor*rgbt_tensor
        tensor1=tensor1+rgb_tensor

        tensor2=t_tensor*rgbt_tensor
        tensor2=tensor2+rgbt_tensor

        tensor3=torch.cat((tensor1,tensor2,rgbt_tensor),dim=1)
        tensor3=self.conv1(tensor3)

        return tensor3
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #两个 1×1×C 的通道描述
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
		# 两层的神经网络,1x1卷积实现通道降维与升维
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes // ratio, 1, bias=False))
        # 再将得到的两个特征相加后经过一个 Sigmoid 激活函数得到权重系数 Mc
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

		# avg_out和max_out 进行拼接cat后为两层，输出为一层
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)     #在通道维度上求平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class ConvBnrelu2d_3(nn.Module):
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
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class DenseBlock(nn.Module):
    def __init__(self,in_channel):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2*in_channel, in_channel, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(2*in_channel, in_channel, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(2*in_channel, in_channel, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2*in_channel, in_channel, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)

    def forward(self, x0):
        x1=self.conv1(x0)
        x1=torch.cat((x0,x1),dim=1)

        x2=self.conv2(x1)
        x2=torch.cat((x0,x2),dim=1)

        x3=self.conv3(x2)
        x3=torch.cat((x0,x3),dim=1)

        x4=self.conv4(x3)
        x4=torch.cat((x0,x4),dim=1)

        x5=self.conv5(x4)
        x5=x5+x0

        x6=self.conv6(x5)
        
        return x6
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class Fusionblock(nn.Module):
    def __init__(self,in_channels):
        super(Fusionblock, self).__init__()
        self.SpatialAttention1=SpatialAttention()
        self.Aspp1=ASPP(in_channels,in_channels)

        self.SpatialAttention2=SpatialAttention()
        self.Aspp2=ASPP(in_channels,in_channels)

        self.DenseBlock=DenseBlock(in_channels)
        self.ChannelAttention=ChannelAttention(in_channels,ratio=16)

        self.conv1= nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv2= nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv3= nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv4= nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    
    def forward(self, rgb_tensor,t_tensor,rgbt_tensor):

        rgbt_tensor_dense = self.DenseBlock(rgbt_tensor)
        rgbt_tensor_ca = self.ChannelAttention(rgbt_tensor_dense)

        rgb_tensor_aspp = self.Aspp1(rgb_tensor)
        rgb_tensor_sa = self.SpatialAttention1(rgb_tensor)
        rgb_a = rgb_tensor_aspp + rgbt_tensor_dense
        rgb_a = rgb_a*rgbt_tensor_ca
        rgb_tensor_out = rgb_a+rgb_tensor_sa
        rgb_tensor_out = self.conv1(rgb_tensor_out)
   

        t_tensor_aspp = self.Aspp2(t_tensor)
        t_tensor_sa = self.SpatialAttention2(t_tensor)
        t_b = t_tensor_aspp + rgbt_tensor_dense
        t_b = t_b*rgbt_tensor_ca
        t_tensor_out = t_b+t_tensor_sa
        t_tensor_out = self.conv2(t_tensor_out)
   

        rgbt_tensor_out = rgb_tensor_aspp + t_tensor_aspp + rgbt_tensor_dense
        rgbt_tensor_out = self.conv3(rgbt_tensor_out)
       

        final_out = rgb_tensor_out+t_tensor_out+rgbt_tensor_out
        final_out = self.conv4(final_out)
    
        return rgb_tensor_out , t_tensor_out , final_out

class DetailCompleModel(nn.Module):
    def __init__(self):
        super(DetailCompleModel, self).__init__()
        self.aspp=ASPP(in_channel=64,depth=64)
        self.conv1=ConvBnrelu2d_3(64,64)
        self.conv2=ConvBnrelu2d_3(128,64)

    def forward(self, detail_tensor,semantic_tensor):
        detail_res = detail_tensor
        semantic_res = semantic_tensor

        semantic_aspp = self.aspp(semantic_tensor)
        plus_tensor = semantic_aspp + detail_tensor
        plus_tensor = self.conv1(plus_tensor)

        semantic_muti=plus_tensor * semantic_res
        detail_muti=plus_tensor * detail_res

        final_detail = detail_res + semantic_muti
        final_semantic = semantic_res + detail_muti

        final_tensor = torch.cat((final_detail,final_semantic),dim=1)
        final_tensor = self.conv2(final_tensor)

        return  final_tensor

class SCPM(nn.Module):
    def __init__(self,tensor_channels):
        super(SCPM, self).__init__()

        self.conv2=nn.Conv2d(tensor_channels, tensor_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False)
        self.CBR=ConvBnrelu2d_3(tensor_channels,tensor_channels)
        self.conv3 = nn.Conv2d(tensor_channels, tensor_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False)
        self.bn   = nn.BatchNorm2d(tensor_channels) 
        self.ASPP =ASPP(tensor_channels,tensor_channels)
        self.conv4=nn.Conv2d(2*tensor_channels, tensor_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, tensor):
        tensor_cat=self.conv2(tensor)

        tensor_res1=tensor_cat
        tensor_cat=self.CBR(tensor_cat)
        tensor_cat=self.conv3(tensor_cat)
        tensor_cat=self.bn(tensor_cat)
        tensor_cat=tensor_cat+tensor_res1
        tensor_cat=F.relu(tensor_cat)

        tensor_res2=tensor_cat
        tensor_aspp=self.ASPP(tensor_cat)
        tensor_final=torch.cat((tensor_aspp,tensor_res2),dim=1)
        tensor_final=self.conv4(tensor_final)

        return tensor_final

class detail_comple(nn.Module):
    def __init__(self):
        super(detail_comple, self).__init__()
        self.down_conv=ConvBnrelu2d_1(64,3)

        self.conv0 = ConvBnrelu2d_1(6,64)
        self.conv1 = ConvBnrelu2d_1(64,64)
       
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)

    def forward(self, RGBT_tensor,RGB_tensor,res0r):
        RGBT_res=RGBT_tensor  #64 480 640 

        RGBT_tensor=self.down_conv(RGBT_tensor)   #64 480 640 ----> 3 480 640 

        RGBT_tensor = torch.cat((RGBT_tensor,RGB_tensor),dim=1)   # 3 + 3
        RGBT_tensor = self.conv0(RGBT_tensor)   #6 480 640 ---> 64 480 640

        res0r=self.up2(res0r)    #64 240 320 ----64 480 640 

        RGBT_tensor = RGBT_tensor+res0r
        RGBT_tensor=self.conv1(RGBT_tensor)  #64 480 640---64 480 640

        return  RGBT_tensor
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class Three_fusion(nn.Module):
    def __init__(self,channel):
        super(Three_fusion, self).__init__()
        self.conv0 = ConvBnrelu2d_3(2*channel,channel)
        self.conv1 = ConvBnrelu2d_3(2*channel,channel)
        self.conv2 = ConvBnrelu2d_3(channel,channel)
        
    def forward(self, RGBT_tensor,RGB_tensor,T_tensor):
        muti1=RGBT_tensor*RGB_tensor
        muti2=RGBT_tensor*T_tensor

        RGB_tensor = RGB_tensor+muti1
        T_tensor = T_tensor+muti2

        RGB_tensor=torch.cat((RGB_tensor,RGBT_tensor),dim=1)
        RGB_tensor=self.conv0(RGB_tensor)

        T_tensor=torch.cat((T_tensor,RGBT_tensor),dim=1)
        T_tensor=self.conv1(T_tensor)

        final=RGB_tensor+T_tensor+RGBT_tensor
        final=self.conv2(final)
        return  final

class Two_fusion(nn.Module):
    def __init__(self,channel,s_channel):
        super(Two_fusion, self).__init__()
        self.conv0 = ConvBnrelu2d_3(channel,channel)
        self.conv1 = ConvBnrelu2d_3(channel,channel)
        self.conv2 = ConvBnrelu2d_3(2*channel,channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(s_channel, s_channel, 1, bias=False), nn.ReLU(),
            nn.Conv2d(s_channel, 2*channel, 1, bias=False))
      
        self.sigmoid = nn.Sigmoid()
        
        self.channel=channel
    def forward(self,RGB_tensor,T_tensor,semantic_tensor):
        avgout = self.sharedMLP(self.avg_pool(semantic_tensor))
        maxout = self.sharedMLP(self.max_pool(semantic_tensor))
        semantic_out = self.sigmoid(avgout + maxout)     # 2 * channel   B C H W
        semantic_rgb = semantic_out[:,0:self.channel,:,:]
        semantic_t = semantic_out[:,self.channel:2*self.channel,:,:]

        rgb_res=RGB_tensor
        t_res=T_tensor

        RGB_tensor = RGB_tensor*semantic_rgb
        RGB_tensor = RGB_tensor+rgb_res
        T_tensor = T_tensor*semantic_t
        T_tensor = T_tensor+t_res

        final_tensor=torch.cat((RGB_tensor,T_tensor),dim=1)
        final_tensor=self.conv2(final_tensor)

        return  final_tensor
#——————————————————————————————————————————————————————————————————————————————————————————————————————
class shallow_fusion(nn.Module):
    def __init__(self,channel):
        super(shallow_fusion, self).__init__()
        self.conv0 = ConvBnrelu2d_3(channel,channel)
     
    def forward(self,RGB_tensor,T_tensor,RGBT_tensor):
        RGB_tensor=RGB_tensor*RGBT_tensor+RGB_tensor
        T_tensor=T_tensor*RGBT_tensor+T_tensor

        final_tensor=RGB_tensor+T_tensor+RGBT_tensor
        final_tensor=self.conv0(final_tensor)
        
        return  final_tensor

class deep_fusion(nn.Module):
    def __init__(self,channel):
        super(deep_fusion, self).__init__()
        self.conv0 = ConvBnrelu2d_3(3*channel,channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=False))

        self.sharedMLP2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=False))
      
        self.sigmoid = nn.Sigmoid()
      
        self.ASPP =ASPP(channel,channel)

    def forward(self,RGB_tensor,T_tensor,RGBT_tensor):
        avgout_rgb = self.sharedMLP1(self.avg_pool(RGB_tensor))
        maxout_rgb = self.sharedMLP1(self.max_pool(RGB_tensor))
        semantic_rgb = self.sigmoid(avgout_rgb + maxout_rgb)     

        avgout_t = self.sharedMLP2(self.avg_pool(T_tensor))
        maxout_t = self.sharedMLP2(self.max_pool(T_tensor))
        semantic_t = self.sigmoid(avgout_t + maxout_t)     
   
        RGB_tensor = RGBT_tensor*semantic_rgb+RGB_tensor
        
        T_tensor = RGBT_tensor*semantic_t+T_tensor
       
        RGBT_tensor=self.ASPP(RGBT_tensor)

        final_tensor=torch.cat((RGB_tensor,T_tensor,RGBT_tensor),dim=1)
        final_tensor=self.conv0(final_tensor)

        return  final_tensor
#————————————————————————————————————————————————————————————————————————————————————————————————————————————
class X_fusion(nn.Module):
    def __init__(self,channel):
        super(X_fusion, self).__init__()
        self.conv0 = ConvBnrelu2d_3(channel,channel)
        self.conv1 = ConvBnrelu2d_3(channel,channel)
        self.conv2 = ConvBnrelu2d_3(2*channel,channel)
        self.conv3 = ConvBnrelu2d_3(channel,channel)
        self.conv4 = ConvBnrelu2d_3(channel,channel)

        self.final_conv=ConvBnrelu2d_3(2*channel,channel)

        self.sigmoid = nn.Sigmoid()

        self.CBR0=ConvBnrelu2d_3(channel,channel)
        self.CBR1=ConvBnrelu2d_3(channel,channel)


    def forward(self,RGB_tensor,T_tensor,RGBT_tensor):
        RGB_tensor = self.conv0(RGB_tensor)
        T_tensor = self.conv1(T_tensor)

        cat_tensor=torch.cat((RGB_tensor,T_tensor),dim=1)
        cat_tensor=self.conv2(cat_tensor)
        cat_tensor_res = cat_tensor
        _,_,H,W=cat_tensor.size()
        cat_tensor = nn.AdaptiveAvgPool2d((H,W))
        cat_tensor = cat_tensor_res - cat_tensor
        cat_tensor_weight=self.sigmoid(cat_tensor)

        RGB_tensor=cat_tensor_weight*RGB_tensor
        T_tensor=cat_tensor_weight*T_tensor

        left_tensor=RGB_tensor+T_tensor

        RGBT_tensor=self.conv3(RGBT_tensor)
        RGBT_res=RGBT_tensor
        RGBT_tensor=self.CBR0(RGBT_tensor)

        left_tensor=self.conv4(left_tensor)
        left_res=left_tensor
        left_tensor=self.CBR1(left_tensor)

        a_tensor=RGBT_tensor*left_res
        b_tensor=RGBT_res*left_tensor

        final_tensor=torch.cat((a_tensor,b_tensor),dim=1)
        final_tensor=self.final_conv(final_tensor)

        return  final_tensor



#————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__=='__main__':
    
    # input_rgb=torch.rand(2,1024,30,40)
    # input_t=torch.rand(2,1024,30,40)
    # input_rgbt=torch.rand(2,1024,30,40)
    # text_net=AFuse(1024)
    # out=text_net(input_rgb,input_t,input_rgbt)
    # print(out.size())

    input_rgb=torch.rand(2,64,240,320)
    input_t=torch.rand(2,64,240,320)
    input_rgbt=torch.rand(2,64,240,320)
    text_net=DetailCompleModel()

    final_out=text_net(input_rgb,input_t)
    print(final_out.size())