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


def accuracyMulti(theta, X, Ytest):
    a = 0
    p = np.dot(X,theta)
    for i in range(len(p)):
       if np.argmax(p[i]) == np.argmax(Ytest[i]):
           a +=1
                
    return a/float(len(Ytest))
    
    
def J(theta,X,Y):
    return np.sum((np.dot(X,theta) - Y)**2)
    
    

def gradientMulti(theta, X, Y):
    return np.dot(X.T,(np.dot(X,theta)- Y))
    

def gradient(theta, X, Y):
    return np.dot((np.dot(X,theta)- Y),X)



def trainingset(actors):
    
    training_input = []
    for actor in actors:
        for i in range(70):
            #print i
            im = rgb2gray(imread('cropped/'+actor+str(i)+'.jpg'))
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            training_input.append(flat)
        
    return training_input
    
def validset(actors):
    
    v_input = []
    for actor in actors:
        for i in range(10):
            #print i
            im = rgb2gray(imread('cropped/'+actor+str(70+i)+'.jpg')) # change 'cropped/' to MvF and 70 to 10 fpr gender classification
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            v_input.append(flat)
        
    return v_input
    
def tset(actors):
    
    t_input = []
    for actor in actors:
        for i in range(10):
            #print i
            im = rgb2gray(imread('cropped/'+actor+str(80+i)+'.jpg')) # change 'cropped/' to MvF and 80 to 20 for gender classification
            flat = im.flatten()/255.0
            flat = np.append(flat,1)
            t_input.append(flat)
        
    return t_input
    


def finiteApproximation(weights, inputs, test_targets, h):
    acc_e = 0
    for p in range(theta.shape[0]):
        for q in range(theta.shape[1]):
            prev = (1.0/(2.0*1025.0))*J(theta, inputs, test_targets)
            df = (1/1025.0)*gradientMulti(theta, inputs, test_targets)
            theta[p][q] += h
            error = ((1.0/(2.0*1025.0))*J(theta, inputs, test_targets) - prev)/h
            acc_e += error - df[p][q]
    return acc_e




def gradientDescent():
    act1 =["baldwin","carell"]
    act2=["baldwin","hader","carell","bracco","gilpin","harmon"]
    vact = ['radcliffe','butler','vartan','chenoweth','drescher','ferrera']

    act = act1 # set to act 2 for gender classification
    valid = act # set to act for baldwin v carell and to vact of gender classification
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
        if iter %5000==0: # to see progress
                print "validation accuracy =", accuracy(theta, valid_input, Vtargets)*100,"%"
                
    '''
    #saving theta 
    map = theta[:-1]
    map.shape = (32,32)
    imsave('20000' + "theta.jpg", map)   #part 4 saves map
    '''
    print "test score =", accuracy(theta, t_input, Ttargets)*100,"%"
    
def gradientDescentMulti():
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
    
    
    for iter in range(10000):
        theta = theta - alpha*gradientMulti(theta, training_input, targets)
        cost = J(theta, training_input, targets)
        if iter %2000==0:
                print "validation accuracy =", accuracyMulti(theta, valid_input, Vtargets)*100,"%"
    '''
    #save images of theta
    map = theta.T
    bald =  (map[0])[:-1]
    bald.shape = (32,32)
    imsave("baldwin"+ "10000" + "theta.jpg", bald)  
    
    had =  (map[1])[:-1]
    had.shape = (32,32)
    imsave("hader"+ "10000" + "theta.jpg", had)    
    
    carell =  (map[2])[:-1]
    carell.shape = (32,32)
    imsave("carell"+ "10000"+ "theta.jpg", carell)   
    
    bracco =  (map[3])[:-1]
    bracco.shape = (32,32)
    imsave("bracco"+ "10000"+ "theta.jpg", bracco)  
    
    gil =  (map[4])[:-1]
    gil.shape = (32,32)
    imsave("gilpin"+ "10000" + "theta.jpg", gil)   
    
    har =  (map[5])[:-1]
    har.shape = (32,32)
    imsave("harmon"+"10000" + "theta.jpg", har)   
    '''
                
    print "test score =", accuracyMulti(theta, t_input, Ttargets)*100,"%"
    
    

def TestFiniteApproximation():
    act=["baldwin","hader","carell","bracco","gilpin","harmon"]
    training_input = np.vstack(trainingset(act))
    
    s=70
    targets = array([[1,0,0,0,0,0]]*s +[[0,1,0,0,0,0]]*s +[[0,0,1,0,0,0]]*s +[[0,0,0,1,0,0]]*s +[[0,0,0,0,1,0]]*s +[[0,0,0,0,0,1]]*s)
    
    im = rgb2gray(imread('cropped/baldwin0.jpg'))
    flat = im.flatten()/255.0
    flat = np.append(flat,1)
    
    theta = array([[0.2]*6]*len(flat))
    h=0.01
    print "for an h of "+str(h)+" the error between the gradient and approximation is"
    print finiteApproximation(theta,training_input,targets,h)
    h=0.0001
    print "for an h of "+str(h)+" the error between the gradient and approximation is"
    print finiteApproximation(theta,training_input,targets,h)
    h=0.000001
    print "for an h of "+str(h)+" the error between the gradient and approximation is"
    print finiteApproximation(theta,training_input,targets,h)
