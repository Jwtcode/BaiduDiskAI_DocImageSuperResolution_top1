import os
from pdb import set_trace as stx
c_path='./c_test'
e_path='./e_test'

c_file_names=os.listdir(c_path)
e_file_names=os.listdir(e_path)

with open ("train.txt",mode="a") as f:
    for name in c_file_names:
        print(name)
      
        f.write('c_test/'+name+' '+str(1))
        f.write("\n")
    for name in e_file_names:
        print(name)
        f.write('e_test/'+name+' '+str(0))
        f.write("\n")
    f.close()
         

# img_path="/home/jiaowt/Downloads/NDATA/chinese_x4/"
# file_names=os.listdir(img_path)
# idx=0
# for name in file_names:
    
#     cur_img_path=os.path.join(img_path,name)
#     x_img=cv2.imread(cur_img_path)
#     h,w,c=x_img.shape
#     x1=0
#     x2=w
#     y1=0
#     y2=h
#     if(h%4 !=0 or w %4!=0):
#         print(name)

#         if(h %4!=0):
#             y_remain=h %4
#             y2=h-y_remain
#         if(w % 4 !=0):
#             x_remain=w % 4
#             x2=w-x_remain
        
#         x_img=x_img[0:y2,0:x2,:]
#         cv2.imwrite(img_path+name,x_img)

#     else:
#         continue

    
        
    
   
   