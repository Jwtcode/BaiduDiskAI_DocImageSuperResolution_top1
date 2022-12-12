import os
from torch.utils.data import Dataset
import cv2
from torchvision import transforms 
from pdb import set_trace as stx
import random
import torch
import numpy as np
random.seed(1234)



class DL(Dataset):
    def __init__(self, rgb_dir, mode='train'):
        super(DL, self).__init__()

        self.mode=mode
        self.trainval_txt="./train.txt"
        with open(self.trainval_txt,mode='r') as f:
             self.info= f.read().splitlines()
             f.close()
        self.transform=transforms.Compose([transforms.ToTensor()])

        self.rgb_dir=rgb_dir
        # self.chinese=os.path.join(rgb_dir,'chinese_n_x')
        # self.english=os.path.join(rgb_dir,'englist_n_x')
        random.shuffle(self.info)
        # numfiles=len(self.info)
        # numtest=numfiles//10
        # numtrain=numfiles-numtest
     
        self.train_txt=self.info
        self.val_txt=self.info
        self.size=(16,16)
      



        if self.mode=='train':
            self.sizex = len(self.train_txt)  # get the size of target
        elif self.mode=='val':
            self.sizex = len(self.val_txt)
        else:
            self.sizex=len(self.val_txt)

       
    def __len__(self):
        return self.sizex

    def __getitem__(self, index):

        if self.mode == 'train':
        
            name,label=self.train_txt[index].split(' ')
            
            # print(name)
            # print (label)

            inp_path = os.path.join(self.rgb_dir,name)
           
            inp_img=cv2.imread(inp_path)
            inp_img=cv2.resize(inp_img,self.size)

            inp_img=self.transform(inp_img)
            label=torch.tensor(float(label))

            sample_batch={"img":inp_img,"label":label}
          
            filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
 
            return  sample_batch,filename

        else :
            name,label=self.val_txt[index].split(' ')

            inp_path = os.path.join(self.rgb_dir,name)
            inp_img=cv2.imread(inp_path)
            inp_img=cv2.resize(inp_img,self.size)
            inp_img=self.transform(inp_img)
            label=torch.tensor(float(label))
            sample_batch={"img":inp_img,"label":label}
            filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
 
            return  sample_batch,filename
       