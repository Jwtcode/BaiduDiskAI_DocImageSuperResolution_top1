import os
import numpy as np
import cv2

image_dict=dict()
def x2_reset_result(result,boundbox,base,h,w,c,retreat,w_split,h_split):

	result= np.transpose(result, (0,2,3,1))
	result=np.clip(result,0,1)
	result=(result*255).astype('uint8')
	reset_img=np.zeros((2*h,2*w,c),dtype='uint8') 
	
	base=base*2
	retreat=retreat*2
	boundbox=boundbox*2
	
	index=0
	#reset_bbox=[]
	reset_img=np.zeros((2*h,2*w,c),dtype='uint8')
	
	for i in range(len(w_split)):
		for j in range(len(h_split)):
			cur_box=boundbox[index]
			x1,x2,y1,y2=cur_box

			
			if(i==0 and j ==0 ):
				reset_img[y1:y2-retreat//2,x1:x2-retreat//2,:]=result[index][0:y2-retreat//2,0:x2-retreat//2,:]
				#reset_bbox.append([x1,x2-retreat//2,y1,y2-retreat//2])
			elif(i==0):
				if(j!=len(h_split)-1):
					reset_img[y1+retreat//2:y2-retreat//2,x1:x2-retreat//2,:]=result[index][0+retreat//2:base-retreat//2,0:base-retreat//2,:]
					#reset_bbox.append([x1,x2-retreat//2,y1+retreat//2,y2-retreat//2])
				else:
					reset_img[y1+retreat//2:y2,x1:x2-retreat//2,:]=result[index][0+retreat//2:base,0:base-retreat//2,:]
					#reset_bbox.append([x1,x2-retreat//2,y1+retreat//2,y2])
			elif(j==0):
				if(i!=len(w_split)-1):
					reset_img[y1:y2-retreat//2,x1+retreat//2:x2-retreat//2,:]=result[index][0:base-retreat//2,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1,y2-retreat//2])
				else:
					reset_img[y1:y2-retreat//2,x1+retreat//2:x2,:]=result[index][0:base-retreat//2,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1,y2-retreat//2])
			else:
				if(i==len(w_split)-1 and j==len(h_split)-1):
					reset_img[y1+retreat//2:y2,x1+retreat//2:x2,:]=result[index][0+retreat//2:base,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1+retreat//2,y2])

				elif(i==len(w_split)-1):
					reset_img[y1+retreat//2:y2-retreat//2,x1+retreat//2:x2,:]=result[index][0+retreat//2:base-retreat//2,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1+retreat//2,y2-retreat//2])
				elif(j==len(h_split)-1):
					reset_img[y1+retreat//2:y2,x1+retreat//2:x2-retreat//2,:]=result[index][0+retreat//2:base,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1+retreat//2,y2])
				else:
					reset_img[y1+retreat//2:y2-retreat//2,x1+retreat//2:x2-retreat//2,:]=result[index][0+retreat//2:base-retreat//2,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1+retreat//2,y2-retreat//2])
			index+=1
	
	return reset_img
	
def x4_reset_result(result,boundbox,base,h,w,c,retreat,w_split,h_split):
	result= np.transpose(result, (0,2,3,1))
	result=np.clip(result,0,1)
	result=(result*255).astype('uint8')
	reset_img=np.zeros((4*h,4*w,c),dtype='uint8')

	base=base*4
	retreat=retreat*4
	boundbox=boundbox*4
	
	index=0
	#reset_bbox=[]
	reset_img=np.zeros((4*h,4*w,c),dtype='uint8')
	
	for i in range(len(w_split)):
		for j in range(len(h_split)):
			cur_box=boundbox[index]
			x1,x2,y1,y2=cur_box

			if(i==0 and j ==0 ):
				reset_img[y1:y2-retreat//2,x1:x2-retreat//2,:]=result[index][0:y2-retreat//2,0:x2-retreat//2,:]
				#reset_bbox.append([x1,x2-retreat//2,y1,y2-retreat//2])
			elif(i==0):
				if(j!=len(h_split)-1):
					reset_img[y1+retreat//2:y2-retreat//2,x1:x2-retreat//2,:]=result[index][0+retreat//2:base-retreat//2,0:base-retreat//2,:]
					#reset_bbox.append([x1,x2-retreat//2,y1+retreat//2,y2-retreat//2])
				else:
					reset_img[y1+retreat//2:y2,x1:x2-retreat//2,:]=result[index][0+retreat//2:base,0:base-retreat//2,:]
					#reset_bbox.append([x1,x2-retreat//2,y1+retreat//2,y2])
			elif(j==0):
				if(i!=len(w_split)-1):
					reset_img[y1:y2-retreat//2,x1+retreat//2:x2-retreat//2,:]=result[index][0:base-retreat//2,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1,y2-retreat//2])
				else:
					reset_img[y1:y2-retreat//2,x1+retreat//2:x2,:]=result[index][0:base-retreat//2,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1,y2-retreat//2])
			else:
				if(i==len(w_split)-1 and j==len(h_split)-1):
					reset_img[y1+retreat//2:y2,x1+retreat//2:x2,:]=result[index][0+retreat//2:base,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1+retreat//2,y2])

				elif(i==len(w_split)-1):
					reset_img[y1+retreat//2:y2-retreat//2,x1+retreat//2:x2,:]=result[index][0+retreat//2:base-retreat//2,0+retreat//2:base,:]
					#reset_bbox.append([x1+retreat//2,x2,y1+retreat//2,y2-retreat//2])
				elif(j==len(h_split)-1):
					reset_img[y1+retreat//2:y2,x1+retreat//2:x2-retreat//2,:]=result[index][0+retreat//2:base,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1+retreat//2,y2])
				else:
					reset_img[y1+retreat//2:y2-retreat//2,x1+retreat//2:x2-retreat//2,:]=result[index][0+retreat//2:base-retreat//2,0+retreat//2:base-retreat//2,:]
					#reset_bbox.append([x1+retreat//2,x2-retreat//2,y1+retreat//2,y2-retreat//2])
			index+=1
	
	return reset_img
	
def class_process(image_path):
	
	x_img = cv2.imread(image_path,1)
	x_img=x_img.astype(np.float32) / 255.0
	image_dict[image_path]=x_img

	class_x_img=cv2.resize(x_img,(16,16))
	class_x_img = np.transpose(class_x_img, (2, 0, 1)).astype(np.float32)
	class_x_img=np.expand_dims(class_x_img,0)

	return class_x_img

def partition(image_path):
		# x_img = cv2.imread(image_path,1)
		# x_img = x_img.astype(np.float32) / 255.0
		x_img=image_dict[image_path]
		base=256
		retreat=24
		w_split=[]
		h_split=[]
		h,w,c=x_img.shape
		###print(x_img.shape)
		dynamic_w=0
		dynamic_h=0
		while(dynamic_w<w):
			w_split.append(dynamic_w)
			dynamic_w+=base-retreat
			if(dynamic_w+base>=w):
				dynamic_w=w-base
				w_split.append(dynamic_w)
				break
		#print(w_split)
	
		
		while(dynamic_h<h):
			h_split.append(dynamic_h)
			dynamic_h+=base-retreat

			if(dynamic_h+base>=h):
				dynamic_h=h-base
				h_split.append(dynamic_h)
				break
		#print(h_split)
		

		boundbox=[]
		for i in range(len(w_split)):
			for j in range(len(h_split)):
				if(i==0 and j ==0 ):
					boundbox.append([base*i,base,base*j,base])
				elif(i==0):
					boundbox.append([base*i,base,h_split[j],h_split[j]+base])
				elif(j==0):
					boundbox.append([w_split[i],w_split[i]+base,base*j,base])
				else:
					boundbox.append([w_split[i],w_split[i]+base,h_split[j],h_split[j]+base])
		##print(boundbox)
		boundbox=np.array(boundbox)
		result=[]
		##print("------------------------------------------------------------------------------------------------------------")
		for box in boundbox:
			cur=x_img[box[2]:box[3],box[0]:box[1],:]
			result.append(cur)
		result=np.stack(result,0)
		result= np.transpose(result, (0,3,1,2))

	
		return result,boundbox,base,h,w,c,retreat,w_split,h_split
