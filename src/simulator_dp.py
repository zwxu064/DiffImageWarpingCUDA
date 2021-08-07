import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
from torch.autograd import Variable
# from multiprocessing import Process, Manager
import operator
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import skimage.transform 

def simdp(RGB_img, depth, image_left, image_right):

    cpsize = 5;

    # # depth_scaled=depth*5.0/(depth.max())

    ker_size=depth

    S=np.shape(depth) #(8, 192, 192)
    b = S[0]
    h = S[1]
    w = S[2]
    # print('start')

    img_left    = torch.zeros([b,3,h,w]).cuda()#np.zeros([b,3,h,w]) # , dtype=torch.long
    # img_left = Variable(img_left).cuda()
    count_left  = torch.zeros([b,1,h,w]).cuda()#np.zeros([b,1,h,w])
    img_right   = torch.zeros([b,3,h,w]).cuda()#np.zeros([b,3,h,w])
    count_right = torch.zeros([b,1,h,w]).cuda()#np.zeros([b,1,h,w])
    # print(RGB_img.size(),img_left.size())
    # ((8, 3, 192, 192), (8, 3, 192, 192))

    #Determining the projected regions on the image plane for every pixel position of the input image/scene       
    for i in range(h):
        for j in range(w):


            # if ker_size[:,i,j]<1:
            #     continue

            y1 = (i - ker_size[:,i,j]).round().int()
            y2 = (i + ker_size[:,i,j]).round().int()
            z1 = j
            z2 = (j + ker_size[:,i,j]).round().int()
            # print('1,',i,j,y1,y2,z1,z2)
            # y1.unsqueeze(1)

            if y1==y2: 
                y2=y2+1

            if z1==z2: 
                z2=z2+1

            y1 = y1.clamp(0, h-1).cpu().numpy()
            y2 = y2.clamp(0, h-1).cpu().numpy()
        
            z2 = z2.clamp(0, w-1).cpu().numpy()
            # print('2',i,j,y1,y2,z1,z2)

            #Synthesizing Left Image    
            img_left[:,:,y1,z1]=img_left[:,:,y1,z1]+RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y2,z1]=img_left[:,:,y2,z1]-RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y1,z2]=img_left[:,:,y1,z2]-RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y2,z2]=img_left[:,:,y2,z2]+RGB_img[:,:,i,j].unsqueeze(2)            

            count_left[:,:,y1,z1]= count_left[:,:,y1,z1]+1
            count_left[:,:,y2,z1]= count_left[:,:,y2,z1]-1
            count_left[:,:,y1,z2]= count_left[:,:,y1,z2]-1
            count_left[:,:,y2,z2]= count_left[:,:,y2,z2]+1

            # z1 = j
            z2 = (j - ker_size[:,i,j]).round().int()

            if z1==z2: 
                z2=z2-1

            # z1 = z1.clamp(0, w-1).cpu().numpy()
            z2 = z2.clamp(0, w-1).cpu().numpy()

            #Synthesizing Right Image                

            img_right[:,:,y1,z1]=img_right[:,:,y1,z1]+RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y2,z1]=img_right[:,:,y2,z1]-RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y1,z2]=img_right[:,:,y1,z2]-RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y2,z2]=img_right[:,:,y2,z2]+RGB_img[:,:,i,j].unsqueeze(2)


            count_right[:,:,y1,z1]= count_right[:,:,y1,z1]+1
            count_right[:,:,y2,z1]= count_right[:,:,y2,z1]-1
            count_right[:,:,y1,z2]= count_right[:,:,y1,z2]-1
            count_right[:,:,y2,z2]= count_right[:,:,y2,z2]+1

    # print(img_left.size(),count_left.size()) #((1, 3, 192, 192), (1, 1, 192, 192))
    integral_image=(cv2.integral(255*img_left.squeeze().cpu().detach().numpy().transpose(1, 2, 0)))
    # print('integral_image', integral_image.shape)
    integral_count=(cv2.integral(255*count_left.squeeze().cpu().detach().numpy()))
    # print('integral_count', integral_count.shape)
    integral_image = torch.from_numpy(integral_image).permute(2, 0, 1).float().unsqueeze(0) / 255
    integral_count = torch.from_numpy(integral_count).unsqueeze(0).unsqueeze(1) / 255
    # print(integral_image.shape,integral_count.shape)
    integral_count = integral_count#.clamp(1e-4, 1)
    # print(integral_count.min())
    im_left = (integral_image/integral_count).clamp(-0.5, 0.5)
 

    integral_image=(cv2.integral(255*img_right.squeeze().cpu().detach().numpy().transpose(1, 2, 0)))
    integral_count=(cv2.integral(255*count_right.squeeze().cpu().detach().numpy()))
    integral_image = torch.from_numpy(integral_image).permute(2, 0, 1).float().unsqueeze(0) / 255
    integral_count = torch.from_numpy(integral_count).unsqueeze(0).unsqueeze(1) / 255
    integral_count = integral_count#.clamp(1e-4, 1)

    im_right = (integral_image/integral_count).clamp(-0.5, 0.5)

    return im_left[:,:,:-1,:-1].cuda(),im_right[:,:,:-1,:-1].cuda()
