#______________________________________________________________________________________________________
import numpy as np
import torch
import random
#随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(1234)
#_______________________________________________________________________________________________________

from setproctitle import setproctitle
import os
setproctitle('lyc ')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   ##################################
import os, argparse, time, sys, stat, shutil
from util.util import calculate_accuracy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn as nn
from util.irseg import IRSeg
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from test2 import TFNet  ###########################################

times = time.time()
local_time = time.localtime(times)

jilupath= r'/data/LYC/code/dragonNet_content/lyc_result_bise3.txt' ##########################################
target='test2'   ##############################################
#_______________________________________________________________________________________________________

# config
n_class   = 9
loss_w = torch.tensor([1.507,16.768,31.767,34.803,39.138,41.331,48.231,46.370,44.147])
def adjust_learning_rate_D(optimizer, epoch):
    lr = args.lr_start * (1-(epoch-1)/500.)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epo, model, train_loader, optimizer, train_dataset):
    model.train() #训练
    adjust_learning_rate_D(optimizer, epo)   #学习率更新策略
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t =time.time()
    cri1 = nn.CrossEntropyLoss(weight = loss_w).cuda() #交叉熵损失
    for it, sample in enumerate(train_loader):
        RGB_images = Variable(sample['image']).cuda()
        T_images = Variable(sample['depth']).cuda()
        labels = Variable(sample['label']).cuda()
        optimizer.zero_grad()
        pre= model(RGB_images, T_images) ################################################
        loss = cri1(pre, labels)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(pre, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)
        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, loss-seg: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(loss), float(acc)))
            t += 5

    content = '| epo:%s/%s train_loss_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(content)

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((n_class, n_class))

    with torch.no_grad():
        for it, sample in enumerate(test_loader):
            RGB_images = Variable(sample['image']).cuda()
            T_images = Variable(sample['depth']).cuda()
            labels = Variable(sample['label']).cuda()
            pre = model(RGB_images, T_images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = pre.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(label, prediction, labels = [0,1,2,3,4,5,6,7,8]) # conf is n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf

            print('|- %s, epo %s/%s. testing iter %s/%s.' % (args.model_name, epo, args.epoch_max, it+1, test_loader.n_iter))

    precision, recall, IoU, = compute_results(conf_total)
    precision, recall, IoU =np.mean(np.nan_to_num(precision)), np.mean(np.nan_to_num(recall)), np.mean(np.nan_to_num(IoU))
    print('the inference is done!')
    print('precision is %s, recall is %s, IoU is %s '%(precision, recall, IoU))
    
    return np.mean(np.nan_to_num(precision)), np.mean(np.nan_to_num(recall)), np.mean(np.nan_to_num(IoU))

def main():

    train_dataset = IRSeg(mode='train', do_aug=True)
    test_dataset = IRSeg(mode='test', do_aug=False)
    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = 1,
        shuffle      = False,
        num_workers = 4,
        pin_memory   = True,
        drop_last    = False
    )
    train_loader.n_iter = len(train_loader)
    test_loader.n_iter = len(test_loader)
    max_R = 0
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' %(args.model_name, epo))

        train(epo, model, train_loader, optimizer, train_dataset)
        #validation(epo, model, val_loader)
        if (epo) % 10 == 0 or (epo) > 150:    
            P, R, I = testing(epo, model, test_loader)   
            '''
            if R > 0.6 and I > 0.55:           
            checkpoint_model_file = os.path.join(r'/data1/zsl_1/VAE_SS/baocun_fusion', 'best'+str(epo)+'.pth')
            print('|- saving check point %s: ' %checkpoint_model_file)
            torch.save(model.state_dict(), checkpoint_model_file) 
            max_R = R 
            '''
            print('OK!')
            jilu = open(jilupath, 'a')
            jilu.write('epoch: ' + str(epo) + '   R: ' + "{:.3f}".format(R) + '  IoU: ' + "{:.3f}".format(I) + '\n')
            jilu.close()
            print('存储位置为',jilupath)
            print(target)
              


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train with pytorch')
    ############################################################################################# 
    parser.add_argument('--model_name',  '-M',  type=str, default='TFNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=2) 
    parser.add_argument('--lr_start',  '-LS',  type=float, default=0.001)
    parser.add_argument('--lr_decay', '-LD', type=float, default=0.95)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=501) # please stop training mannully
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1) 
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    parser.add_argument('--target', default=target , type=str,help='model.target')
    args = parser.parse_args()
                 
    model = eval(args.model_name)()
    
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    
    print('training %s on GPU with pytorch' % (args.model_name))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('存储位置为',jilupath)
    print(target)
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))

    str0=args.target
    jilu = open(jilupath, 'a')
    jilu.write('———————————————————————'+str0+'—————————————————————————' + time.strftime("%Y-%m-%d %H:%M:%S",local_time)+'\n')
    jilu.close()

    main()
