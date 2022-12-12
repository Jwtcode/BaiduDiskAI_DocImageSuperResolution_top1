import os
from option import args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from dataloader import DL
from resnet18 import VGG16
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import warnings
warnings.filterwarnings('ignore')  

######### Set Seeds ###########
def evaluator(model):
    return test(model, device, criterion, val_loader)

def train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=True):

    for epoch in range(start_epoch, args.epochs + 1):

        if TRAIN:
            model.train()

            for i, (sample_batch,_) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                img=sample_batch["img"]
                label=sample_batch["label"]
                img = img.cuda()
                label = label.cuda()
                logit=model(img)
                loss = l1loss(logit.view(-1),label)
                loss.backward()
                optimizer.step()

        #### Evaluation ####
        
        model.eval()
        correct=0
        total=0
        for ii, (sample_batch,_) in enumerate((val_loader), 0):
            img=sample_batch["img"]
            label=sample_batch["label"]
            img = img.cuda()
            label = label.cuda()
            logit=model(img)
            loss = l1loss(logit.view(-1),label)
            logit=logit.detach().view(-1).cpu().numpy()
            label = label.cpu().numpy()
            pred = np.where(logit>=0.5,1,0)
            correct += len(np.where(pred==label)[0])
            total+=len(logit)

        acc=correct/total
        print(
        'Epoch: [%d]' 
        "[val_accuracy: %.3f]" % (epoch,acc))
        if TRAIN ==False:
            break

        model_dict_name='model_best.pth'
        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,model_dict_name))

        scheduler.step()
        model.train()

if __name__=="__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1

    result_dir = "./result"
    model_dir  = "./checkpoint"

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = args.dir_data
    val_dir   = args.dir_data

    ######### Model ###########
    model = VGG16()
    model.cuda()

    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


    ######### Scheduler ###########
    warmup_epochs = 1
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-warmup_epochs, eta_min=0)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if args.resume:
        path_chk_rest    = utils.get_last_path(model_dir, 'best.pth')
    
        utils.load_checkpoint(model,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch+1):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
    #     print('------------------------------------------------------------------------------')
    #     print("==> Resuming Training with learning rate:", new_lr)
    #     print('------------------------------------------------------------------------------')
    ######### Loss ###########
    l1loss = nn.BCELoss()
    ######### DataLoaders ###########
    train_dataset = DL(train_dir,mode='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

    val_dataset = DL(val_dir,mode='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=True)
    #############################剪枝####################################
    #stx()
    # path_chk_rest="./checkpoint/74_1.0.pth"
    # utils.load_checkpoint(model,path_chk_rest)
    # start_epoch = utils.load_start_epoch(path_chk_rest)
    # utils.load_optim(optimizer, path_chk_rest)
    # for i in range(1, start_epoch):
    #     scheduler.step()
    # new_lr = scheduler.get_lr()[0]
    # # 打印一下看看
    # summary(model, (3, 64, 64), device="cuda")
    # # 输出一下精度
    # print("----------------原模型精度-----------------")
    # train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=False)
    # # 定义剪枝配置
    # config_list = [{'sparsity': 0.6, 'op_types': ['Conv2d']}]
 
    # pruner = L1FilterPruner(model, config_list)
    # model = pruner.compress()
    # pruner.export_model(model_path="./checkpoint/prune.pth", mask_path="./checkpoint/mask.pth")
    # # 压缩模型
    # pruner._unwrap_model()
    # x=torch.randn(1,3,64,64,device="cuda:0")
    # m_Speedup = ModelSpeedup(model, x, "./checkpoint/mask.pth", "cuda")
    # m_Speedup.speedup_model()
    # # 打印一下模型
    # summary(model, (3, 64, 64), device="cuda")
    # # 打印一下模型精度
    # print("---------------剪枝模型精度------------------")
    # train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=False)
    # # 再次训练微调模型
    # train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=True)
    # # 打印一下精度
    # print("---------------剪枝微调精度------------------")
    # train(start_epoch,args,model,optimizer,scheduler,l1loss,train_loader,val_loader,TRAIN=False)
    # # 保存模型
    # torch.save(model.state_dict(), "./checkpoint/prune_model.pth")






