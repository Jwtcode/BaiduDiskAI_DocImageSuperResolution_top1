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
from model.rcan import RCAN
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import warnings
warnings.filterwarnings('ignore')  

######### Set Seeds ###########
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
model = RCAN(args)
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
    path_chk_rest    = utils.get_last_path(model_dir, 'base.pth')
  
    utils.load_checkpoint(model,path_chk_rest)
#     start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

#     for i in range(1, start_epoch):
#         scheduler.step()
#   new_lr = scheduler.get_lr()[0]
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:", new_lr)
#     print('------------------------------------------------------------------------------')
######### Loss ###########
l1loss = nn.L1Loss()

######### DataLoaders ###########
train_dataset = DL(train_dir, {'patch_size':args.patch_size},scale=args.scale,mode='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=26, shuffle=True, num_workers=6, drop_last=False, pin_memory=True)

val_dataset = DL(val_dir, {'patch_size':args.patch_size},scale=args.scale,mode='val')
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=6, drop_last=False, pin_memory=True)

#print('===> Start Epoch {} End Epoch {}'.format(start_epoch,args.epochs + 1))
#print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, args.epochs + 1):
    epoch_start_time = time.time()
    train_id = 1
    
    model.train()
    
    for i, (lr,x2_hr,x4_hr,_) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        lr = lr.cuda()
        x2_hr = x2_hr.cuda()
        x4_hr = x4_hr.cuda()
        x2_sr,x4_sr = model(lr)
      
        loss = l1loss(x2_sr, x2_hr)+l1loss(x4_sr, x4_hr)
        loss.backward()
        optimizer.step()


      
    #### Evaluation ####
        if (i+1) % (len(train_loader)//3) == 0:
            model.eval()
            psnr_val_rgb_x2 = []
            psnr_val_rgb_x4 = []
            for ii, (lr, x2_hr,x4_hr, _) in enumerate((val_loader), 0):
                #print("%d/%d"%(ii,len(val_loader)))
                lr=lr.cuda()
                x2_hr = x2_hr.cuda()
                x4_hr = x4_hr.cuda()
                with torch.no_grad():
                    x2_sr,x4_sr = model(lr)
                for res,tar in zip(x2_sr,x2_hr):
                    psnr_val_rgb_x2.append(utils.torchPSNR(res, tar))
                for res,tar in zip(x4_sr,x4_hr):
                    psnr_val_rgb_x4.append(utils.torchPSNR(res, tar))

            psnr_val_rgb  = 0.4*torch.stack(psnr_val_rgb_x2).mean().item()+0.6*torch.stack(psnr_val_rgb_x4).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            #print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}_psnr_{psnr_val_rgb}.pth")) 

            scheduler.step()
            model.train()

