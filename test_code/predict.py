import os
import sys
import glob
import cv2
import numpy as np
from DataProcess import *
import onnxruntime
from pdb import set_trace as stx

batch_size=30
def part_c_e(src_image_dir):
  sess0 = onnxruntime.InferenceSession(path_or_bytes='./vgg.onnx', providers=['CUDAExecutionProvider'])
  input_name0=sess0.get_inputs()[0].name
  output_name0=sess0.get_outputs()[0].name

  c_image_paths=[]
  e_image_paths=[]

  image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
  for image_path in image_paths:

    class_x_img=class_process(image_path)
    class_res= sess0.run([output_name0], {input_name0:class_x_img})[0][0]
    
    if(class_res>0.5):
      c_image_paths.append(image_path)
    else:
      e_image_paths.append(image_path)
  
  return c_image_paths,e_image_paths
  
def c_process(c_image_paths,save_dir1,save_dir2):
  c_sess = onnxruntime.InferenceSession(path_or_bytes='./c.onnx', providers=['CUDAExecutionProvider'])
  input_name1 = c_sess.get_inputs()[0].name
  output_name1 = c_sess.get_outputs()[0].name
  output_name2 = c_sess.get_outputs()[1].name

  for image_path in c_image_paths:
     
        save_path1 = os.path.join(save_dir1, os.path.basename(image_path))
        save_path2 = os.path.join(save_dir2, os.path.basename(image_path))
        inp,boundbox,base,h,w,c,retreat,w_split,h_split=partition(image_path)

        if(inp.shape[0]>batch_size):
          total=inp.shape[0]
          batch=total // batch_size
          final_batch_size=total % batch_size
          x2_list=[]
          x4_list=[]
          for i in range (batch+1):
            if(i==batch and final_batch_size!=0):
              cur_batch_inp=inp[i*batch_size:]
              x2_res,x4_res= c_sess.run([output_name1,output_name2], {input_name1:cur_batch_inp})
              x2_list.append(x2_res)
              x4_list.append(x4_res)
              break
            cur_batch_inp=inp[i*batch_size:(i+1)*batch_size]
            x2_res,x4_res= c_sess.run([output_name1,output_name2], {input_name1:cur_batch_inp})
            x2_list.append(x2_res)
            x4_list.append(x4_res)

          if(len(x2_list)==2):
            x2_res=np.concatenate((x2_list[0],x2_list[1]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1]),0)

          elif(len(x2_list)==3):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2]),0)
          elif(len(x2_list)==4):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2],x2_list[3]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2],x4_list[3]),0)
          elif(len(x2_list)==5):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2],x2_list[3],x2_list[4]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2],x4_list[3],x4_list[4]),0)
        else:
            x2_res,x4_res= c_sess.run([output_name1,output_name2], {input_name1:inp})

        x2_img=x2_reset_result(x2_res,boundbox,base,h,w,c,retreat,w_split,h_split)
        x4_img=x4_reset_result(x4_res,boundbox,base,h,w,c,retreat,w_split,h_split)
        cv2.imwrite(save_path1,x2_img)
        cv2.imwrite(save_path2,x4_img)

def e_process(e_image_paths,save_dir1,save_dir2):
  e_sess = onnxruntime.InferenceSession(path_or_bytes='./e.onnx', providers=['CUDAExecutionProvider'])
  input_name1 = e_sess.get_inputs()[0].name
  output_name1 = e_sess.get_outputs()[0].name
  output_name2 = e_sess.get_outputs()[1].name

  for image_path in e_image_paths:

        save_path1 = os.path.join(save_dir1, os.path.basename(image_path))
        save_path2 = os.path.join(save_dir2, os.path.basename(image_path))
        inp,boundbox,base,h,w,c,retreat,w_split,h_split=partition(image_path)

        if(inp.shape[0]>batch_size):
          total=inp.shape[0]
          batch=total // batch_size
          final_batch_size=total % batch_size
          x2_list=[]
          x4_list=[]
          for i in range (batch+1):
            if(i==batch and final_batch_size!=0):
              cur_batch_inp=inp[i*batch_size:]
              x2_res,x4_res= e_sess.run([output_name1,output_name2], {input_name1:cur_batch_inp})
              x2_list.append(x2_res)
              x4_list.append(x4_res)
              break
            cur_batch_inp=inp[i*batch_size:(i+1)*batch_size]
            x2_res,x4_res= e_sess.run([output_name1,output_name2], {input_name1:cur_batch_inp})
            x2_list.append(x2_res)
            x4_list.append(x4_res)

          if(len(x2_list)==2):
            x2_res=np.concatenate((x2_list[0],x2_list[1]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1]),0)

          elif(len(x2_list)==3):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2]),0)
          elif(len(x2_list)==4):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2],x2_list[3]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2],x4_list[3]),0)
          elif(len(x2_list)==5):
            x2_res=np.concatenate((x2_list[0],x2_list[1],x2_list[2],x2_list[3],x2_list[4]),0)
            x4_res=np.concatenate((x4_list[0],x4_list[1],x4_list[2],x4_list[3],x4_list[4]),0)
        else:
            x2_res,x4_res= e_sess.run([output_name1,output_name2], {input_name1:inp})

        x2_img=x2_reset_result(x2_res,boundbox,base,h,w,c,retreat,w_split,h_split)
        x4_img=x4_reset_result(x4_res,boundbox,base,h,w,c,retreat,w_split,h_split)
        cv2.imwrite(save_path1,x2_img)
        cv2.imwrite(save_path2,x4_img)


if __name__ == "__main__":
    assert len(sys.argv) == 4
  
    src_image_dir = sys.argv[1]
    pred_x2_dir = sys.argv[2]
    pred_x4_dir = sys.argv[3]

    # src_image_dir="./test"
    # pred_x2_dir='./result2'
    # pred_x4_dir='./result4'
    

    if not os.path.exists(pred_x2_dir):
      os.makedirs(pred_x2_dir)
    if not os.path.exists(pred_x4_dir):
      os.makedirs(pred_x4_dir)
    c_image_paths,e_image_paths=part_c_e(src_image_dir)
    c_process(c_image_paths, pred_x2_dir, pred_x4_dir)
    e_process(e_image_paths, pred_x2_dir, pred_x4_dir)