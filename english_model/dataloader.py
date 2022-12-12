import os
from torch.utils.data import Dataset
import cv2
from torchvision import transforms 
from pdb import set_trace as stx
import random
import numpy as np
random.seed(1234)

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 15
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            w, h = imgs[i].shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            imgs[i] = cv2.warpAffine(imgs[i], rotation_matrix, (h, w))
    return imgs

class DL(Dataset):
    def __init__(self, rgb_dir, img_options=None,scale=4,mode='train'):
        super(DL, self).__init__()

        self.mode=mode
        self.trainval_txt="./train.txt"

        with open(self.trainval_txt,mode='r') as f:
            self.info= f.read().splitlines()
        f.close()
        #transforms.Normalize((0.9049, 0.9042, 0.9006), (0.2090, 0.2086, 0.2165))
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.scale=scale
        self.lr_path=os.path.join(rgb_dir,'english_n_x')
        self.hr_x2_path=os.path.join(rgb_dir,'english_n_x2')
        self.hr_x4_path=os.path.join(rgb_dir,'english_n_x4')
        random.shuffle(self.info)
        numfiles=len(self.info)
        numtest=numfiles//10
        numtrain=numfiles-numtest
     
        self.train_txt=self.info[:numtrain]
        self.val_txt=self.info[numtrain:]
      

        self.img_options = img_options

        if self.mode=='train':
            self.sizex = len(self.train_txt)  # get the size of target
        elif self.mode=='val':
            self.sizex = len(self.val_txt)
        else:
            self.sizex=len(self.val_txt)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):

        if self.mode == 'train':
        
            name,pos=self.train_txt[index].split(' ')
            pos=pos.split(',')
            inp_path = os.path.join(self.lr_path,name)
            x2_tar_path = os.path.join(self.hr_x2_path,name)
            x4_tar_path = os.path.join(self.hr_x4_path,name)
         
          
            inp_img = cv2.imread(inp_path)
            x2_tar_img = cv2.imread(x2_tar_path)
            x4_tar_img = cv2.imread(x4_tar_path)
            # print((int(pos[0]),int(pos[1]),int(pos[2]),int(pos[3])))

            inp_img=inp_img[int(pos[0]):int(pos[1]),int(pos[2]):int(pos[3]),:]
            x2_tar_img=x2_tar_img[int(pos[0])*2:int(pos[1])*2,int(pos[2])*2:int(pos[3])*2,:]
            x4_tar_img=x4_tar_img[int(pos[0])*4:int(pos[1])*4,int(pos[2])*4:int(pos[3])*4,:]
            
            ############################################
#             w4,h4,c4=x4_tar_img.shape
#             if(w4!=512 or h4 !=512):
#                 print(x4_tar_img.shape)
#                 print(pos)
#                 print (name)
#             all_input = [inp_img,x2_tar_img,x4_tar_img] 
#             all_input = random_rotate(all_input)
#             inp_img = all_input[0]
#             x2_tar_img = all_input[1]
#             x4_tar_img = all_input[2]

            inp_img=self.transform(inp_img)
            x2_tar_img=self.transform(x2_tar_img)
            x4_tar_img=self.transform(x4_tar_img)

            filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
 
            return  inp_img,x2_tar_img,x4_tar_img, filename

        else :
            name,pos=self.val_txt[index].split(' ')
            pos=pos.split(',')

            inp_path = os.path.join(self.lr_path,name)
            x2_tar_path = os.path.join(self.hr_x2_path,name)
            x4_tar_path = os.path.join(self.hr_x4_path,name)
            # print(inp_path)
            # print(tar_path)
            inp_img = cv2.imread(inp_path)
            x2_tar_img = cv2.imread(x2_tar_path)
            x4_tar_img = cv2.imread(x4_tar_path)

            inp_img=inp_img[int(pos[0]):int(pos[1]),int(pos[2]):int(pos[3]),:]
            x2_tar_img=x2_tar_img[int(pos[0])*2:int(pos[1])*2,int(pos[2])*2:int(pos[3])*2,:]
            x4_tar_img=x4_tar_img[int(pos[0])*4:int(pos[1])*4,int(pos[2])*4:int(pos[3])*4,:]
            inp_img=self.transform(inp_img)
            x2_tar_img=self.transform(x2_tar_img)
            x4_tar_img=self.transform(x4_tar_img)

            filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
            return  inp_img,x2_tar_img, x4_tar_img,filename
       