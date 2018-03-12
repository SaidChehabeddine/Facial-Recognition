from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


'''
balwin 0 - 69 are training
carell 0 - 69 are training
baldwin 70 -80 are validation
carell  70-80 are validation
'''

## Part 3: Gradient Descent to determine 0 parameters between Baldwin and Carell
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    
    This Function was given by Michael Guerzhoy	
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.



def accuracy(theta, X, Ytest):
    a = 0
    p = np.dot(X,theta)
    for i in range(len(p)):
       if np.argmax(p[i]) == np.argmax(Ytest[i]):
           a +=1
                
    return a/float(len(Ytest))


def J(theta,X,Y):
    return np.sum((np.dot(X,theta) - Y)**2)
    
    

def gradient(theta, X, Y):
    return np.dot(X.T,(np.dot(X,theta)- Y))



def trainingset(actors):
    
    training_input = []
    for actor in actors:
        for i in range(70):
            print i
            im = rgb2gray(imread('cropped/'+actor+str(i)+'.jpg'))
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            training_input.append(flat)
        
    return training_input
    
def validset(actors):
    
    v_input = []
    for actor in actors:
        for i in range(10):
            print i
            im = rgb2gray(imread('cropped/'+actor+str(70+i)+'.jpg'))
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            v_input.append(flat)
        
    return v_input
    
def tset(actors):
    
    t_input = []
    for actor in actors:
        for i in range(10):
            print i
            im = rgb2gray(imread('cropped/'+actor+str(80+i)+'.jpg'))
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            t_input.append(flat)
        
    return t_input
    
def gradientDescent():
    act=["baldwin","hader","carell","bracco","gilpin","harmon"]
    training_input = np.vstack(trainingset(act))
    valid_input = np.vstack(validset(act))
    t_input = np.vstack(tset(act))
    
    s=70
    targets = array([[1,0,0,0,0,0]]*s +[[0,1,0,0,0,0]]*s +[[0,0,1,0,0,0]]*s +[[0,0,0,1,0,0]]*s +[[0,0,0,0,1,0]]*s +[[0,0,0,0,0,1]]*s)
    v=10
    Vtargets = array([[1,0,0,0,0,0]]*v +[[0,1,0,0,0,0]]*v +[[0,0,1,0,0,0]]*v +[[0,0,0,1,0,0]]*v +[[0,0,0,0,1,0]]*v +[[0,0,0,0,0,1]]*v)
    Ttargets = array([[1,0,0,0,0,0]]*v +[[0,1,0,0,0,0]]*v +[[0,0,1,0,0,0]]*v +[[0,0,0,1,0,0]]*v +[[0,0,0,0,1,0]]*v +[[0,0,0,0,0,1]]*v)

    alpha = 0.001
    
    im = rgb2gray(imread('cropped/baldwin0.jpg'))
    flat = im.flatten()/255.0
    flat = np.append(flat,1)
    
    theta = array([[0]*6]*len(flat))
    cost = J(theta,training_input,targets)
    
    
    for iter in range(13000):
        theta = theta - alpha*gradient(theta, training_input, targets)
        cost = J(theta, training_input, targets)
        if iter %10000==0:
                print "validation accuracy =", accuracy(theta, valid_input, Vtargets)*100,"%"
                '''
                map = theta.T
                bald =  (map[0])[:-1]
                bald.shape = (32,32)
                imsave("baldwin"+ str(iter) + "theta.jpg", bald)  
                
                had =  (map[1])[:-1]
                had.shape = (32,32)
                imsave("hader"+ str(iter) + "theta.jpg", had)    
                
                carell =  (map[2])[:-1]
                carell.shape = (32,32)
                imsave("carell"+ str(iter) + "theta.jpg", carell)   
                
                bracco =  (map[3])[:-1]
                bracco.shape = (32,32)
                imsave("bracco"+ str(iter) + "theta.jpg", bracco)  
                
                gil =  (map[4])[:-1]
                gil.shape = (32,32)
                imsave("gilpin"+ str(iter) + "theta.jpg", gil)   
                
                har =  (map[5])[:-1]
                har.shape = (32,32)
                imsave("harmon"+ str(iter) + "theta.jpg", har)   
                '''
                

    print "test score =", accuracy(theta, t_input, Ttargets)*100,"%"
