# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 03:20:38 2019

@author: olahs
"""
import os
import numpy as np
import glob 
import matplotlib.pyplot as plt 
from scipy.misc import *
from PIL import Image
#M x N
#xs= np.loadtxt("data_3d.txt",delimiter=" ", skiprows=1, usecols=(0,1,2))
#print(xs.shape)
# print xs
 

# reduce image resolution because of computational power for the covariance and eigen decomposition
img_a = 60
img_b = 70

# loads a greyscale version of every jpg image in the directory.
# INPUT  : directory
# OUTPUT : imgs - n x p array (n images with p pixels each)
#def load_images(directory):
#    imgs = []
#    # get a list of all the picture filenames
#    jpgs = glob.glob(directory + '/*.jpg')
#    # load a greyscale version of each image
#    for i in jpgs:
#        img = cv2.imread(i, 0)
#        imgn = np.array(cv2.resize(img,(img_a, img_b))).flatten()
#        imgs.append(imgn)  
#    return imgs
 

def load_images(directory):
    ## create buffer for the rgb bands
    imgs_r1 = []
    imgs_g1 = []
    imgs_b1 = []
    # get a list of all the picture filenames
    jpgs = glob.glob(directory + '/*.jpg')
    # load a greyscale version of each image
    for i in jpgs:
        imgw = Image.open(i)
        imgw = imgw.convert('RGB')
        
        ## get the RGB bands for the image
        rr,gg,bb = imgw.split() 
        
        ## resize the image to reduce computation
        r1 = rr.resize((img_a, img_b), Image.ANTIALIAS)
        g1 = gg.resize((img_a, img_b), Image.ANTIALIAS)
        b1 = bb.resize((img_a, img_b), Image.ANTIALIAS)
        ## convert each rgb image to rgb vectors
        imgn_r1 = np.array(r1).flatten()
        imgn_g1 = np.array(g1).flatten()
        imgn_b1 = np.array(b1).flatten()
        ## append the rgb vector into a matrix RGB
        imgs_r1.append(imgn_r1)  
        imgs_g1.append(imgn_g1)  
        imgs_b1.append(imgn_b1)  
    return imgs_r1, imgs_g1, imgs_b1

def save_image(out_dir, subdir, img_id, img_dims, data):
    directory = out_dir + "/" + subdir
    if not os.path.exists(directory): os.makedirs(directory)
    imsave(directory + "/image_" + str(img_id) + ".jpg", data)

    
imgs_r1, imgs_g1, imgs_b1 = load_images(r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/bw_imgRGB')
xs_r1 = np.array(imgs_r1)
xs_g1 = np.array(imgs_g1)
xs_b1 = np.array(imgs_b1)


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
    k=5
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
#    Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
    print(data-rec) # test reconstruction error
    
    return pr, rec, mean

pr_r1, rec_r1, mean_r1 = pca(xs_r1)
pr_g1, rec_g1, mean_g1 = pca(xs_g1)
pr_b1, rec_b1, mean_b1 = pca(xs_b1)


##################################################
###################################################

img_dims = (img_b,img_a)  #change the dimension back before of the initial transpose

rgbArray = np.zeros((img_b,img_a,3), 'uint8')
rgbArrayMean = np.zeros((img_b,img_a,3), 'uint8')
outpath = r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/OutputRGB'

mean_rn1 = np.reshape(mean_r1,(img_b, img_a))
mean_gn1 = np.reshape(mean_g1,(img_b, img_a))
mean_bn1 = np.reshape(mean_b1,(img_b, img_a))
rgbArrayMean[..., 0] = mean_rn1 
rgbArrayMean[..., 1] = mean_gn1
rgbArrayMean[..., 2] = mean_bn1
imsave(outpath + "/mean.jpg", rgbArrayMean)




# save each RGB eigenface as an image
rgbArray = np.zeros((img_b,img_a,3), 'uint8')
for i in range(rec_r1.shape[1]):
    #select the columns in the projected faces for each band
    ef_r = rec_r1[:, i]
    ef_g = rec_g1[:, i]
    ef_b = rec_b1[:, i]
    #select the real part for each band 
    rgbArray[..., 0] = np.reshape(ef_r.real,(img_b, img_a)) 
    rgbArray[..., 1] = np.reshape(ef_g.real,(img_b, img_a))  
    rgbArray[..., 2] = np.reshape(ef_b.real,(img_b, img_a))  
    save_image(outpath, "eigenfaces", i, img_dims, rgbArray)
    

rgbArrayrec = np.zeros((img_b,img_a,3), 'uint8')
# save the reconstructed faces of each subject
for i in range(rec_r1.shape[1]):
    ## #first get the columns (subjects) in the projected faces for red
    J_r1 = rec_r1[:,i]
    Jn_r1 = J_r1 + mean_r1 # add the mean face to the face projection of the subject to get the recon
    Bn_r1 = np.reshape(Jn_r1.real,(img_b, img_a)) # reshape the column back to a 2D matrix
    new_B_r1 = Image.fromarray(Bn_r1)
    new_B_r1 = new_B_r1.convert("L")
    
    ## #first get the columns (subjects) in the projected faces for green
    J_g1 = rec_g1[:,i]
    Jn_g1 = J_g1 + mean_g1 
    Bn_g1 = np.reshape(Jn_g1.real,(img_b, img_a))
    new_B_g1 = Image.fromarray(Bn_g1)
    new_B_g1 = new_B_g1.convert("L")
    ## #first get the columns (subjects) in the projected faces for blue
    J_b1 = rec_b1[:,i]
    Jn_b1 = J_b1 + mean_b1
    Bn_b1 = np.reshape(Jn_b1.real,(img_b, img_a))
    new_B_b1 = Image.fromarray(Bn_b1)
    new_B_b1 = new_B_b1.convert("L")
    
    ## now combine the bands
    rgbArrayrec[..., 0] = new_B_r1
    rgbArrayrec[..., 1] = new_B_g1
    rgbArrayrec[..., 2] = new_B_b1
    
    save_image(outpath, "OutputRGB", i, img_dims, rgbArrayrec)

plt.imshow(rgbArrayrec)
plt.title('reconsRGB')
plt.show()
