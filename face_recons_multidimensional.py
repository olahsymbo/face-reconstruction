# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:49:40 2019

@author: olahs
"""

import os
import inspect

app_path = inspect.getfile(inspect.currentframe())
projec_dir = os.path.realpath(os.path.dirname(app_path))
import numpy as np
import glob 
import matplotlib.pyplot as plt 
from scipy.misc import *
from PIL import Image
 

# reduce image resolution because of computational power for the covariance and eigen decomposition
img_a = 60
img_b = 70

def load_images(directory):
    imgs = []
    # get a list of all the picture filenames
    jpgs = glob.glob(directory + '/*.jpg')
    # load a greyscale version of each image
    for i in jpgs:
        img = Image.open(i) 
#        img = rgb2gray(img1)
#        img = cv2.imread(i, 0)
        imgn = img.resize((img_a, img_b), Image.ANTIALIAS)
        imgn = np.array(imgn).flatten()
        imgs.append(imgn)  
    return imgs

def save_image(out_dir, subdir, img_id, img_dims, data):
    directory = out_dir + "/" + str(subdir)
    if not os.path.exists(directory): os.makedirs(directory)
    imsave(directory + "/image_" + str(img_id) + ".jpg", data)

    
xs = load_images(os.path.join(projec_dir, 'face_reconstruction/bw_img'))
xs = np.array(xs)

# sort and get k largest eigenvalues
k = np.arange(0,1000,20)
## We start codding PCA here
def pca(xs, k):

    #get mean
    mean= np.mean(xs,axis=0)

    #N x M
    data= (xs-mean).T # we need transpose 
    print(data.shape)
    # print data

    #N x N
    covData=np.cov(data) # calculate covariance matrix
    print(covData.shape)

    eigenvalues, eigenvectors = np.linalg.eig(covData)
    print(eigenvalues.shape) # N long
    print(eigenvectors.shape) # N x N
    print(eigenvalues)
    print(eigenvectors)

    idx = eigenvalues.argsort()[-k:][::-1]
    print(idx)
    
    eigenvalues = eigenvalues[idx] # k long 
    eigenvectors = eigenvectors[:,idx] # N x k
    print(eigenvalues.shape)
    print(eigenvectors.shape)
    print(eigenvalues)
    print(eigenvectors)
    
    #projection and reconstruction
    pr= np.dot(data.T,eigenvectors) # (M N) * (N k) => (M k)
    rec= np.dot(eigenvectors, pr.T) #(N k) * (k M) => (N M)
    print(data-rec) # test reconstruction error
    
    return pr, rec, mean


##################################################
###################################################

img_dims = (img_b,img_a)  #change the dimension back before of the initial transpose
#
outpath = os.path.join(projec_dir, 'face_reconstruction/all_recons')
outeigen = os.path.join(projec_dir, 'face_reconstruction/eigenf')
outmean = os.path.join(projec_dir, 'face_reconstruction/meann')
#

# save the reconstructed faces of each subject
for ii in range(xs.shape[0]):
    
    for d in range(len(k)):
        pr, rec, mean = pca(xs, k[d])
                 
        J = rec[:,ii] #first get the columns (subjects) in the projected faces
        Jn = J + mean  # add the mean face to the face projection of the subject to get the reconstructed face
        Bn = np.reshape(Jn.real,(img_b, img_a)) # reshape the column back to a 2D matrix
        new_B = Image.fromarray(Bn) # convert the matrix (array) to image
        new_B = new_B.convert("L") # convert it back to grayscale (just to be sure it is in grayscale)
        save_image(outpath, ii, k[d], img_dims, new_B)
        save_image(outeigen, ii, k[d], img_dims, J.reshape(img_b, img_a).real)
#        filename=os.path.join(outpath+"/img_"+"%d.png" % i)
#        new_B.save(filename) # save the reconstructed face
        save_image(outmean, k[d], k[d], img_dims, mean.reshape(img_b, img_a))
# lets view the face reconstruction of the last subject
plt.imshow(new_B,cmap='gray')
plt.title('recons')
plt.show()
