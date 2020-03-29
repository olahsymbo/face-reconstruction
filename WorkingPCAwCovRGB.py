# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 02:33:30 2019

@author: olahs
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 06:41:16 2019

@author: olahs
"""

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
img_a = 80
img_b = 90

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
    directory = out_dir + "/" + str(subdir)
    if not os.path.exists(directory): os.makedirs(directory)
    imsave(directory + "/image_" + str(img_id) + ".jpg", data)

    
imgs_r1, imgs_g1, imgs_b1 = load_images(r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/bw_imgRGB')
xs_r1 = np.array(imgs_r1)
xs_g1 = np.array(imgs_g1)
xs_b1 = np.array(imgs_b1)
 
## We start codding PCA here
k = np.arange(1, 102, 1)


## We start codding PCA here
def pca(xs, k):
    # get mean
    mean = np.mean(xs, axis=0)
    # print mean.shape
    # print mean

    # N x M
    data = (xs - mean).T  # we need transpose
    #print(data.shape)
    #print(data)

    # N x N
    #covData = np.cov(data)  # calculate covariance matrix
    #print(covData.shape)

    #eigenvalues, eigenvectors = np.linalg.eig(covData)
    #print(eigenvalues.shape)  # N long
    #print(eigenvectors.shape)  # N x N
    #print(eigenvalues)
    U, Sigma, VT = linalg.svd(data.transpose(), full_matrices=False)
    
    print("data:", data.shape)
    print("U:", U.shape)
    print("Sigma:", Sigma.shape)
    print("V^T:", VT.shape)
    
#    idx = sigma.argsort()[-k:][::-1]
    #print(idx)
    eigenvectors = VT[:k,:].T
#    sigma = sigma[k]  # k long
#    eigenvectors1 = eigenvectors[:, k]  # N x k
    #print(eigenvalues.shape)
    #print(eigenvectors.shape)
    #print(eigenvalues)

#    pr = np.dot(data, eigenvectors1)  # (M N) * (N k) => (M k)
    pr = np.dot(data.T, eigenvectors)
    print("pr:", pr.shape)
    # projection and reconstruction
    rec = np.dot(pr, eigenvectors.T)  # (N k) * (k M) => (N M)
#    print(data - rec)  # test reconstruction error
    
    return eigenvectors, rec, mean


##################################################
###################################################

img_dims = (img_b,img_a)  #change the dimension back before of the initial transpose


#
outpath = r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/Outt/all_reconsRGB'
outeigen = r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/Outt/eigenfRGB'
outmean = r'C:/Users/olahs/Documents/Python_scripts/pca_reconsRGB/Outt/meannRGB'
#



## save each eigenface as an image
#for i in range(rec.shape[1]):
#    ef = rec[:, i] #select the columns in the projected faces
#    save_image(outpath, "eigenfaces", i, img_dims, ef.real) # select on the real part from the complex matrix
#    

rgbArrayrec = np.zeros((img_b,img_a,3), 'uint8')
Jj = np.zeros((img_b,img_a,3), 'uint8')
rgbArrayMean = np.zeros((img_b,img_a,3), 'uint8')


# save the reconstructed faces of each subject
for ii in range(xs_r1.shape[0]):
    
    for d in range(len(k)):
        #pr, rec, mean = pca(xs, k[d])
        
        eigenvectors_r1, rec_r1, mean_r1 = pca(xs_r1, k[d])
        eigenvectors_g1, rec_g1, mean_g1 = pca(xs_g1, k[d])
        eigenvectors_b1, rec_b1, mean_b1 = pca(xs_b1, k[d])
        
        mean_rn1 = np.reshape(mean_r1,(img_b, img_a))
        mean_gn1 = np.reshape(mean_g1,(img_b, img_a))
        mean_bn1 = np.reshape(mean_b1,(img_b, img_a))
        
        J_r = rec_r1[ii,:] #first get the columns (subjects) in the projected faces
        J_r1 = np.reshape(J_r.real,(img_b, img_a))
        Jn_r = J_r + mean_r1  # add the mean face to the face projection of the subject to get the reconstructed face
        Bn_r = np.reshape(Jn_r.real,(img_b, img_a)) # reshape the column back to a 2D matrix
        new_B_r = Image.fromarray(Bn_r) # convert the matrix (array) to image
        new_B_r = new_B_r.convert("L") # convert it back to grayscale (just to be sure it is in grayscale)
        
        J_g = rec_g1[ii,:] #first get the columns (subjects) in the projected faces
        J_g1 = np.reshape(J_g.real,(img_b, img_a))
        Jn_g = J_g + mean_g1  # add the mean face to the face projection of the subject to get the reconstructed face
        Bn_g = np.reshape(Jn_g.real,(img_b, img_a)) # reshape the column back to a 2D matrix
        new_B_g = Image.fromarray(Bn_g) # convert the matrix (array) to image
        new_B_g = new_B_g.convert("L") # convert it back to grayscale (just to be sure it is in grayscale)

        J_b = rec_b1[ii,:] #first get the columns (subjects) in the projected faces
        J_b1 = np.reshape(J_b.real,(img_b, img_a))
        Jn_b = J_b + mean_b1  # add the mean face to the face projection of the subject to get the reconstructed face
        Bn_b = np.reshape(Jn_b.real,(img_b, img_a)) # reshape the column back to a 2D matrix
        new_B_b = Image.fromarray(Bn_b) # convert the matrix (array) to image
        new_B_b = new_B_b.convert("L") # convert it back to grayscale (just to be sure it is in grayscale)

        Jj[..., 0] = J_r1
        Jj[..., 1] = J_g1
        Jj[..., 2] = J_b1
        
        rgbArrayrec[..., 0] = new_B_r
        rgbArrayrec[..., 1] = new_B_g
        rgbArrayrec[..., 2] = new_B_b
        
        rgbArrayMean[..., 0] = mean_rn1 
        rgbArrayMean[..., 1] = mean_gn1
        rgbArrayMean[..., 2] = mean_bn1

        save_image(outpath, ii, k[d], img_dims, rgbArrayrec)
        
        save_image(outeigen, ii, k[d], img_dims, Jj)
#        filename=os.path.join(outpath+"/img_"+"%d.png" % i)
#        new_B.save(filename) # save the reconstructed face
        save_image(outmean, k[d], k[d], img_dims, rgbArrayMean)
# lets view the face reconstruction of the last subject

# lets view the face reconstruction of the last subject
# save each eigenface as an image
    rgbArrayEigenV = np.zeros((img_b,img_a,3), 'uint8')
    for i in range(eigenvectors_r1.shape[1]):
        EnR = np.reshape(eigenvectors_r1[:, i], (img_b, img_a))
        EnR1 = Image.fromarray(EnR * 255) # convert the matrix (array) to image
         
        EnG = np.reshape(eigenvectors_g1[:, i], (img_b, img_a))
        EnG1 = Image.fromarray(EnG * 255) # convert the matrix (array) to image
         
        EnB = np.reshape(eigenvectors_b1[:, i], (img_b, img_a))
        EnB1 = Image.fromarray(EnB * 255) # convert the matrix (array) to image
         
        rgbArrayEigenV[..., 0] = EnR1 
        rgbArrayEigenV[..., 1] = EnG1
        rgbArrayEigenV[..., 2] = EnB1       
        
        save_image(outeigen, "eigenfaces", i, img_dims, rgbArrayEigenV)
        
        
plt.imshow(rgbArrayrec,cmap='gray')
plt.title('recons')
plt.show()