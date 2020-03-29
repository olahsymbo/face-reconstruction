# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:04:22 2019

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
        imgn = img.resize((img_a, img_b), Image.ANTIALIAS)
        imgn = np.array(imgn).flatten()
        imgs.append(imgn)  
    return imgs

def save_image(out_dir, subdir, img_id, img_dims, data):
    directory = out_dir + "/" + subdir
    if not os.path.exists(directory): os.makedirs(directory)
    imsave(directory + "/image_" + str(img_id) + ".jpg", data.reshape(img_dims))

    
xs = load_images(os.path.join(projec_dir, "face_reconstruction/bw_img"))
xs = np.array(xs)

## We start codding PCA here
def pca(xs):
    

    #get mean
    mean= np.mean(xs,axis=0)
    # print mean.shape
    # print mean

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

    #sort and get k largest eigenvalues
    k=1000
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


pr, rec, mean = pca(xs)


img_dims = (img_b,img_a)  #change the dimension back before of the initial transpose

outpath = r'C:/Users/olahs/Documents/Python_scripts/face_reconstruction/Output3'

imsave(outpath + "/mean.jpg", mean.reshape(img_b, img_a))

# save each eigenface as an image
for i in range(rec.shape[1]):
    ef = rec[:, i] #select the columns in the projected faces
    save_image(outpath, "eigenfaces", i, img_dims, ef.real) # select on the real part from the complex matrix
    
# save the reconstructed faces of each subject
for i in range(rec.shape[1]):
    
    J = rec[:,i] #first get the columns (subjects) in the projected faces
    Jn = J + mean  # add the mean face to the face projection of the subject to get the reconstructed face
    Bn = np.reshape(Jn.real,(img_b, img_a)) # reshape the column back to a 2D matrix
    new_B = Image.fromarray(Bn) # convert the matrix (array) to image
    new_B = new_B.convert("L") # convert it back to grayscale (just to be sure it is in grayscale)
    filename=os.path.join(outpath+"/img_"+"%d.png" % i)
    new_B.save(filename) # save the reconstructed face

# lets view the face reconstruction of the last subject
plt.imshow(new_B,cmap='gray')
plt.title('recons')
plt.show()
