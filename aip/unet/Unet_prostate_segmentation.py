#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:21:21 2019

Advanced Image Processing Exercises 2019
Week 3: Machine learning and Pattern Recognition
Exercise from Lecture 4: Unet for image segmentation
@author: Jose Castillo
In this exercise, you will learn:   
   *) 
   *) How to write code to build a unet using the deep learning package Keras.
   *) How to present data to your network and train it.
   *) Hot the learning of the networks occurs and how to quantify it.
   

During this exercise you will work with some examples fro    
    


"""

#%% 
"""
For this exercise we will use part of a public data set named PROSTATEX, 
this data set was used in a challange to classify prostate cancer tumor accord_
ing different gleasons grade. However, for our learning goals , we will use it
to train a network that will learn how to do automatic prostate segmentation.

If you want to know more details about this data set you can consult the 
following link 

https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges

Description: 
The PROSTATEx Challenge ("SPIE-AAPM-NCI Prostate MR Classification Challenge”) 
focused on quantitative image analysis methods for the diagnostic classification 
of clinically significant prostate cancers and was held in conjunction with the 
2017 SPIE Medical Imaging Symposium (see http://www.spie.org/PROSTATEx/).  PROSTATEx 
ran from November 21, 2016 to January 15, 2017, though a "live" version has also 
been established at https://prostatex.grand-challenge.org which serves as an ongoing
 way for researchers to benchmark their performance for this task.   

"""

#%%   
import glob
import os
import numpy as np
from sklearn.model_selection import ShuffleSplit
import SimpleITK as sitk
import matplotlib.pyplot as plt


#%% Excercise  Getting to know the data:

"""Excercise  Getting to know the data:
   Let's get a feeling of the data first. If you check the patient's folder
   you will find two folders; images and their corresponding segmentations.
   Each image correspond to a 2D prostate axial slice. Use the SimpleITK 
   package to load a prostate slice from any 
   patient and the segmentation belonging to that slice, preferebly choose
   for one of the slices of the middle section,for instance: if the 
   prostate has 14 slices, load slice 7 or one close to it. After that, plot 1 figure 
   showing 3 following images:
   -The t2 sequence, 
   -The mask 
   -The segmentation overlay on the t2 / The segmentation contour (your choice).
    Hint: To show the overlay you may use any function learne during previous
    exercises. As suggestion , you may choose one of these atributes on pyplot.contour 
    or modifying for the color map to plt.cm.viridis. 
 """
 
 
t2_ima= sitk.ReadImage('/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed/ProstateX-0100/images/Px_ProstateX-0100_slice_7.nii')
se_ima= sitk.ReadImage('/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed/ProstateX-0100/segmentations/Px_ProstateX-0100_slice_7.nii')

t2_arr= sitk.GetArrayFromImage(t2_ima)
se_arr= sitk.GetArrayFromImage(se_ima)

plt.figure(),plt.subplot(1,3,1),plt.imshow(t2_arr,cmap='gray'),plt.subplot(1,3,2),
plt.imshow(se_arr,cmap='gray'),plt.subplot(1,3,3),plt.imshow(t2_arr,cmap='gray'),
plt.contour(se_arr,alpha=0.5)
plt.close()

plt.figure(),plt.subplot(1,3,1),plt.imshow(t2_arr,cmap='gray'),plt.subplot(1,3,2),
plt.imshow(se_arr,cmap='gray'),plt.subplot(1,3,3),plt.imshow(t2_arr,cmap='gray'),
plt.imshow(se_arr,cmap=plt.cm.viridis,alpha=.3)
plt.close()
#%%
#t2_ima= sitk.ReadImage('/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed/ProstateX-0100/images/Px_ProstateX-0100_slice_7.nii')
#
#t2_arr= sitk.GetArrayFromImage(t2_ima)
#plt.figure(),plt.subplot(1,3,1),plt.imshow(t2_arr,cmap='gray'),plt.subplot(1,3,2),
#plt.imshow(se_arr,cmap='gray'),plt.subplot(1,3,3),plt.imshow(t2_arr,cmap='gray'),
#plt.contour(se_arr,alpha=0.5)
#
#
#t2_arr= t2_arr.astype('float32')
#plt.figure(),plt.subplot(1,3,1),plt.imshow(percentile_norm(t2_arr),cmap='gray'),plt.subplot(1,3,2),
#plt.imshow(se_arr,cmap='gray'),plt.subplot(1,3,3),plt.imshow(t2_arr,cmap='gray'),
#plt.imshow(se_arr,cmap=plt.cm.viridis,alpha=.3)

#%% Example Improving how you find your data:
 
#     DON'T NEED TO CODE, ONLY READ, UNDERSTAND and RUN it  
 
""" Example improving how you load your data:
   You may have noticed that the data is structured in a specific way. This last 
was done  not only with the objective to keep the data organized, but also to 
make the data proccesing with python easier. On the following example
you will se how to get advantage of the directory names to locate the images
for patient 100. In this case we are going to use the python module called 
"glob", Run the following lines, and understand what  the code is doing.    
"""
# 
# First we define our patient (px) data folder
px       = 'ProstateX-0100'
px_fol   = '/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed' 


# We define to variables to save the image and segmentation path
# as you can see that  glob list from a specific directory (px_fol) ,
# from a specific patient (px) , all the files (*) that ends as nifti format (nii.)
img_path = glob.glob(px_fol + '/'+px+'/images/*.nii')
seg_path = glob.glob(px_fol + '/'+px+'/segmentations/*.nii')

"""Check what contains the two variables: img_path and seg_path

"""


#%% Exercise  2: 
"""Write a function called create_px_path that takes a list of patients as input. Then, the function 
    should return  two lists. One list should contain the path
    to the images and the other the paths to the segmentation files.
    IMPORTANT: check that the odering inside the two list corresponds to the same 
    image and segmentation. For instance, if the first element in the images list  
    corresponds to patient 100 slice 8 .Then,the first element of the segmentation 
    list should correspond to patient 100, slice 8 as well.
 """
def create_px_path(px_list):
    
    px_fol   = '/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed' 
    
    x_imag = [] 
    y_segm = []
    
    for p in px_list:
        images  = glob.glob(px_fol + '/'+p+'/images/*.nii')
        segmen  = glob.glob(px_fol+ '/'+p+'/segmentations/*.nii')
        x_imag  = x_imag+images 
        y_segm  = y_segm+segmen

    return x_imag, y_segm

#%%  Exercise, Data variability:
    
np.random.seed(180)   #Do not modify this line, it is for reproducibility.
"""Example ,Data variability: 
     One problem regarding MR data, is the variability of intensities in the image.
     Each patient can greatly vary from one to another.Let's have a look on this issue:
    -Used the provided list of patients to load 6 prostate slices(zero segmentations)and plot 
     them in a 2x3 figure. 
    -Answer the following:
        -observe the images intensities between common anatomical structures, for instance
         the high value on the the peripheral zone, also look to the
         dark intensities in the trainsition zone.
      -Do you see differences between signal intensities?, 
      - Take a moment and thinkg about optimization learning algorithms,similar 
        to what you learned on registation week. Answer the following:
       -How do you think these differences in signal intensity might affect the 
        learning process of the neural network or any other machine learning method?
       
       "Your Answer:         "  
 """

images,_ = create_px_path(os.listdir(px_fol)) 

#plt.figure()
#for i in range(2):
#   ran_int = np.random.randint(1,np.size(images))
#   im_arra = sitk.ReadImage(images[ran_int])
#   print(images[ran_int])
#   im_arra = sitk.GetArrayFromImage(im_arra)
#   plt.subplot(1,2,i+1),plt.imshow(im_arra,cmap='gray')
# 
# 
 
 #%%  Exercise data Normalization.
np.random.seed(180)   #Do not modify this line , it is for reproducibility.
"""Exercise data Normalization:
    In order to reduce the data differences , let's perform a data normalization
    using the following formula: 
       norm_imag = image - percentile_10 / (percentile_90 - percentile_10)
    -Therefore, define a percentile_norm function that takes an image array as
     input and return the image data normalized using the previous formula.
     HINT
     Read the np.percentile function documentation to know how to obtain the 90th
     and 10th percentile.
     -Plot the images after being normalized and observe the differences using 
      again 2x3 figure size as the previous exercise.
"""

def percentile_norm(datas):
    
    x_90  = np.percentile(datas,95)
    x_10  = np.percentile(datas,5)
    datas -=  x_10  
    datas  /= (x_90 - x_10)
    
    return datas
 
#   
#plt.figure()
#for i in range(3):
#   ran_int = np.random.randint(1,np.size(images))
#   im_arra = sitk.ReadImage(images[ran_int])   
#   print(images[ran_int])
#   im_arra = sitk.GetArrayFromImage(im_arra)
#   plt.subplot(2,3,i+1),plt.imshow(im_arra,cmap='gray')
#   im_arra = im_arra.astype('float32')
#   im_arra = percentile_norm(im_arra)
#   plt.subplot(2,3,i+4),plt.imshow(im_arra,cmap='gray')   

 
#plt.figure(),plt.subplot(2,2,1),plt.imshow(t2_a2,cmap='gray'),plt.subplot(2,2,2),
#plt.imshow(t2_a2_n,cmap='gray'),plt.subplot(2,2,3),plt.imshow(t2_a,cmap='gray'),
#plt.subplot(2,2,4),plt.imshow(t2_a_n,cmap='gray')


#%%  Example , splitting the data
#     DON'T NEED TO CODE ONLY READ, UNDERSTAND and RUN it .
    
"""
# Splitting the data:
  In this part of the exercise we have a function that will split our patient
  data in training, validation and test. The function takes a list of patient
  names adn randomy split it. First in training and test, then the training
  is split in traning and validation. In each interation we take 20% of the patient
  data. The function returns a data dictionary with the names of the patient
  on each set. 
"""
def split_data_train_val_test(px_fol):

    patients = os.listdir(px_fol)
    patients = np.asarray(patients)

    ss    = ShuffleSplit(n_splits=1,test_size=0.20)
    ss.get_n_splits(patients)
    for train_index, test_index in ss.split(patients):
        xt, x_test = patients[train_index], patients[test_index]
        
    ss = ShuffleSplit(n_splits=1, test_size=0.20)
    ss.get_n_splits(xt)    
    for ten_index, val_index in ss.split(xt):
        x_train_in, x_val_in= xt[ten_index], xt[val_index]       
   
    px_splits = {'train': np.ndarray.tolist(x_train_in),
                 'val'  : np.ndarray.tolist(x_val_in)  ,
                 'test' : np.ndarray.tolist(x_test)    }
    
    return px_splits

px_fol_path = '/media/data/Prostate_data_sets/processed_data_set/prostatex_test_balint_cod/processed'
px_split_di = split_data_train_val_test(px_fol_path)

 
#%% 

"""While training the network we will need to provide the patients data paths,
   now that we have our data divided in sets. Use your create_px_path function from
   exercise 2 to generate the paths of each set. You will have to provide an image
   dictionary per set (x_train/val/test) and a segmentation dictionary(y_train /val/test).
   We use the x and y as a convention in machine learning for training data (x) and target/label data
   (y).
"""
    

x_train , y_train = create_px_path(px_split_di['train'])
x_valid , y_valid = create_px_path(px_split_di['val'])
x_test  , y_test  = create_px_path(px_split_di['test'])

#%% Utility Functions


def generate_batch_norm(batch):
    data = []
    
    for img in batch:
        #print('img = '+str(img))
        img_data = sitk.ReadImage(img) 
                
        img_data = sitk.GetArrayFromImage(img_data)
        img_data = img_data.astype('float32')
        img_data = percentile_norm(img_data)
        data.append(img_data)

    data = np.stack(data)


    data = np.reshape(data, (data.shape[0],data.shape[2],data.shape[1],1))
    return data
 
def generate_batch(batch):
    data = []
    
    for img in batch:
        #print('img = '+str(img))
        img_data = sitk.ReadImage(img) 
                
        img_data = sitk.GetArrayFromImage(img_data)
        img_data = img_data.astype('float32')
        data.append(img_data)

    data = np.stack(data)


    data = np.reshape(data, (data.shape[0],data.shape[2],data.shape[1],1))
    return data   

#%% Example Data generator:
#   DO NOT MODIFY 
 
"""DO NOT MODIFY
   Example Data generator:
   This function is made to generate a image batch. Neural networks consume a lot
   of computing resources, therefore to train a network like U-net we feed it with
   smaller portions of data, a data batch. This function takes a list of patient
   image paths , image targets and the desired batch size. As you can see, the
   end of the function finish with the word "yield" instead of return. This means 
   that the local variables will be keept while the condition of the function is
   true, this kind of statement is used to return intermediate results. 
   In other words, we can keep "feeding" the Unet with data continuosly, by providing
   data batches until we reach to the end of the image list (while 
   the function is called by the network). 
""" 
def data_generator(x_lis,y_tar,d_size):
    while True:
        len_lis = len(x_lis)
        nu_part = (len_lis//d_size)+1
        count   = 0
        
        for i in range(nu_part):
        
            if count >= len_lis:
                continue    
            
            if i+1 == d_size:
               p_list = x_lis[count:]
               segment= y_tar[count:]
            else:
               p_list  = x_lis[count:count+d_size]
               segment = y_tar[count:count+d_size]
            
            images = generate_batch_norm(p_list)
            target = generate_batch(segment)
            count += d_size 
            yield images, target  
            
            
#%%
            
            
#import matplotlib.pyplot as plt
#bach_norm_x=x_train[0:5]
#bach_norm_x= generate_batch_norm(bach_norm_x)
#bach_norm_y=y_train[0:5]
#bach_norm_y= generate_batch_norm(bach_norm_y)
#plt.imshow(bach_norm_x[2,:,:],cmap='gray'),plt.imshow(bach_norm_y[2,:,:],alpha=0.25)
#%%
"""DO NOT MODIFY
   Do not modify
   Dice coeficcent formula: 
   To compute the error during learning process we will use the dice 
   coefficent. Every time the network performs a prediction (y_pred= segmentation prediction),
   this new predicted image will be compared with the ground truth (y_true). Then, 
   based on this error value, the network will update its parameters in order to improve 
   the prediction for the next time it sees an example. This process occurs mathematically
   using differential equations. Therfore, we need define the dice function using tensors,
   which allow us to compute the derivatives in python.
"""
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    K.print_tensor(intersection, message="Dice intersection:")
    return -((2. * intersection + K.epsilon()) / (K.sum(y_true_f)
                                                  + K.sum(y_pred_f)
                                                  + K.epsilon()))
    
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)    
#%%
"""Exercise Define Unet using keras:
   On this part you will have to complete the code to later define unet. Given
   that the orginal unet request many computatinal resources, we will define
   a slighly simpler unet. As you can see most of the code is written already, you
   will need to use the image provided for this practical to complete the code.
   what you need to do make it work is:
   -Define the input layer accodring to your image size and color channels
    (height, with, channels)     
   -For the convolutional layers define the number of kernels and a kernel 
    size 3. Notice that the kernel size is doubled on each layer while the kernel
    size is kept.
    HINT: you may use keras documentation to help you with the code:
       https://keras.io/layers/convolutional/
    -Define the Max pooling layers using a pooling of 2x2
    -When the model is finished, run the line create_unet. Should not return
     any error message.      

"""
from keras import Input
from keras.models import Model
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.layers import Concatenate, Conv2D
from keras.preprocessing.image import array_to_img
from keras.callbacks import ModelCheckpoint


from keras.utils import plot_model
    
def create_unet():
    '''
    Creates a U-Net
    '''
    print('Creating U-Net...')

    # First, we have to provide the dimensions of the input images
    inputs = Input((192, 192,1))

    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print('conv1 shape:', conv1.shape)
    print('pool1 shape:', pool1.shape)

    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('conv2 shape:', conv2.shape)
    print('pool2 shape:', pool2.shape)

    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    drop4 = Dropout(0.5)(conv3)  # Added
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print('conv3 shape:', conv3.shape)
    print('pool3 shape:', pool3.shape)

    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    print('conv4 shape:', conv4.shape)


    up7 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(drop4))  # Changed
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    print('conv7 shape:', conv7.shape)

    up8 = Conv2D(16, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    print('conv8 shape:', conv8.shape)

    up9 = Conv2D(16, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    print('conv9 shape:', conv9.shape)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print('conv10 shape:', conv10.shape)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=dice_coef, metrics=[dice_coef_loss])

    #model.load_weights('/home/jose/Documents/Prostate_radiomics/Unet_AIP/weights2')  # Load the pre-trained U-Net



    print('Got U-Net!')

    return model

model = create_unet()# Create the U-Net
#plot_model(model,'/home/jose/Documents/Prostate_radiomics/Unet_AIP/model_exercise.png'
#           ,show_shapes=True)
#%% Example: Training Unet

"""Example: Training Unet
   In this section we start training Unet. First we define a number of epochs, 
   which is  the number of times that Unet is going to "observe"
   the whole training set, and learn from it the features to perform the segmentation.
   We also define a batch size  and a history variable. In history we are going
   to save all the training process. 
   
   Keras package allow us to train the model using the attribute "fit_generator", 
   we give as input the data generator previously defined, we also define what 
   data is going to be used for training and validation.
   
   Run this part of the code, each epoch should take approximately 3 seconds,
   the code will run for 300 epochs. So you can take this tame to have a short
   coffe brake. After the training is complete, continue with the next exercise.
"""
filepath='/home/jose/Documents/Prostate_radiomics/Unet_AIP/weights_new/weights.{epoch:02d}_'+'.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='dice_coef_loss',period=25, verbose=1, save_best_only=True,
                              mode='min')
callbacks_list = [checkpoint]

epch = 300
batch_size = 4    
history=model.fit_generator(data_generator(x_train,y_train, 
                            batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epch,
            validation_data=data_generator(x_valid,y_valid,
                                           batch_size),
            validation_steps=len(x_valid) // batch_size,
            callbacks=callbacks_list)    

#%%
            
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H:%M:%S") 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()
plt.savefig('/home/jose/Documents/Prostate_radiomics/Unet_AIPmodel_loss_'+str(date_time)+'.png')
plt.close()
