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
    
    This function was provided by Michael Guerzhoy
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.



def accuracy(theta, X, Ytest):
    a = 0
    p = np.dot(X,theta)
    for i in range(len(p)):
        if ((abs(p[i] - 1) < abs(p[i] + 1))):
            if(Ytest[i] ==1):		
                a += 1
        else:
            if(Ytest[i] == -1):		
                a += 1
                
    return a/float(len(Ytest))


def J(theta,X,Y):
    '''
    This is the cost function J(0)
    '''
    return np.sum((np.dot(X,theta) - Y)**2)
    
    
    

def gradient(theta, X, Y):
    return np.dot((np.dot(X,theta)- Y),X)



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
            im = rgb2gray(imread('MvF/'+actor+str(10+i)+'.jpg')) # change cropped to MvF and 70 to 10
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            v_input.append(flat)
        
    return v_input
    
def tset(actors):
    
    t_input = []
    for actor in actors:
        for i in range(10):
            print i
            im = rgb2gray(imread('MvF/'+actor+str(20+i)+'.jpg')) # change cropped to MvF and 80 to 20
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            t_input.append(flat)
        
    return t_input
    
def gradientDescent():
    act1 =["baldwin","carell"]
    act2=["baldwin","hader","carell","bracco","gilpin","harmon"]
    vact = ['radcliffe','butler','vartan','chenoweth','drescher','ferrera']

    act = act2 # set to act 2 for gender classification
    valid = vact # set to act for baldwin v carell and to vact of gender classification
    training_input = np.vstack(trainingset(act))
    valid_input = np.vstack(validset(valid))
    t_input = np.vstack(tset(valid))
    
    targets = np.append(np.ones(70*len(act)/2),-1*np.ones(70*len(act)/2))
    Vtargets = np.append(np.ones(10*len(act)/2),-1*np.ones(10*len(act)/2))
    Ttargets = np.append(np.ones(10*len(act)/2),-1*np.ones(10*len(act)/2))

    alpha = 0.0001                # 0.01 for Alec v Carell 0.0001 for gender classification
    im = rgb2gray(imread('cropped/baldwin0.jpg'))
    flat = im.flatten()/255.0
    flat = np.append(flat,1)
    
    theta = 0*np.ones(len(flat))
    cost = J(theta,training_input,targets)
    
    
    for iter in range(20000):
        theta -= alpha*gradient(theta, training_input, targets)
        cost = J(theta, training_input, targets)
        if iter %5000==0:
                print "validation accuracy =", accuracy(theta, valid_input, Vtargets)*100,"%"
                '''
                map = theta[:-1]
                map.shape = (32,32)
                imsave(str(iter) + "theta.jpg", map)   #part 4 saves map
                '''

    print "test score =", accuracy(theta, t_input, Ttargets)*100,"%"
