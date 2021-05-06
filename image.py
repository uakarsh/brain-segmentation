from tensorflow import keras
import tensorflow as tf
import numpy as np
from skimage import io
import skimage
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
from tensorflow.keras.models import model_from_json as mj
import skimage.transform
import cv2
import matplotlib.pyplot as plt
epsilon = 1e-5
smooth = 1


import os
modelFile = 'clf-resnet-model.json'
model = mj(open(modelFile).read())
model.load_weights(os.path.join(os.path.dirname(modelFile), 'clf-brain.hdf5'))


modelFile = 'ResUNet-seg-model.json'
seg_model = mj(open(modelFile).read())
seg_model.load_weights(os.path.join(os.path.dirname(modelFile), 'seg_model_brain.hdf5'))
#print("The model summary for segmentation model is:",seg_model.summary())
#target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown','silence']

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def prediction(a):
        image = io.imread(a)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image = skimage.transform.resize(image=image,output_shape=(256,256,image.shape[2]))
        original_image = image
        #print("The shape of the image is:",original_image.shape)
        image = np.expand_dims(image,axis = 0) 
        predict =  model.predict(image)
        seg_pred = seg_model.predict(image)
        seg_pred = np.array(seg_pred).squeeze().round()
        original_image[seg_pred==1] = (255,0,0)
        # print(seg_pred.max())
        if np.argmax(predict):
            pred = np.array(seg_pred).squeeze().round()
            cv2.imshow("Mask",pred)
            cv2.imshow("Masked image",original_image)
            #cv2.waitKey()
            cv2.destroyAllWindows()
            return "It has some tumour spotted"
        else:
            return 'No tumour found, looks like you are safe:)'
        return 1




#print("*"*50)
# x = 'TCGA_DU_7013_19860523_14.tif'
# y = 'TCGA_CS_4941_19960909_15.tif'

# print("The prediction for x is:",prediction(x))
# print("The prediction for y is:",prediction(y))
# file = 'audio/yes.wav'
# sample,samples_rate = librosa.load(file)
# samples = librosa.resample(sample,samples_rate,8000)
# # samples = np.expand_dims(samples,axis=0).reshape(-1,1)
# #print("The shape of the sample is:",samples.shape)
# print("The keyword is:",prediction(file))