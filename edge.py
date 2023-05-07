import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sci



def edge(img, sigma = 2, t_low = .2, t_high = .5):
    
    
    
    
    ## Computing gradient
    # Input parameters
    width = int(np.ceil(4*sigma))
    
    # Creating Gaussian kernel
    H = np.zeros(width)
    for i in range(0,int(width)):
        x = i - ((width-1)/2)
        H[i] = np.exp(-(x**2)/(2*sigma**2))
            
    H = H*(1/(np.sqrt(2*np.pi)*sigma))
    H = H/np.sum(H) # ensuring the kernel sums up to zero
      
    # Creating 1D DoG kernel
    DoG = np.gradient(H)
    DoG[np.sign(DoG) == -1] = DoG[np.sign(DoG) == -1]/-np.sum(DoG[np.sign(DoG) == -1]) # ensuring the kernel sums up to zero
    DoG[np.sign(DoG) == +1] = DoG[np.sign(DoG) == +1]/np.sum(DoG[np.sign(DoG) == +1]) # ensuring the kernel sums up to zero
    
    # Computing gradient via convolution
    # Reshaping the 1D filter into a 2D matrix with a single column
    DoGx = np.reshape(DoG, (1, -1))
    DoGy = np.reshape(DoG, (-1, 1))
    # Performing 1D convolution along rows
    dx = sci.signal.convolve2d(img, DoGx, mode='same', boundary= "symm") # x component
    dy = -sci.signal.convolve2d(img, DoGy, mode='same', boundary= "symm") # y component
    D = np.sqrt(dx**2+dy**2)  # 2D gradient norm
    D = D/np.max(D) # Normalized 2D gradient
    
    # Angle with respect to the axis of the burner in degrees
    theta = np.arctan2(dx,dy)
    theta = np.abs(theta*180/np.pi)
    
    
    
 
    ## Non-maxima suppression
    # Defining max range
    y_max, x_max = D.shape
    
    # Defining final thin image
    nms = np.ones(D.shape)
    nms[0:,0] = 0
    nms[0:,-1] = 0
    nms[0,0:] = 0
    nms[-1,0:] = 0
    
    mask = np.zeros((3,3))
    
    for j in range(1,y_max-1):
        for i in range(1,x_max-1):
            mask = D[j-1:j+2,i-1:i+2]
            if 0<=theta[j,i]<22.5 or 157.5<=theta[j,i]<=180:
                if mask[0,1]>D[j,i] or mask[2,1]>D[j,i]:
                    nms[j,i] = 0
            elif 22.5<=theta[j,i]<67.5:
                if mask[0,2]>D[j,i] or mask[2,0]>D[j,i]:
                    nms[j,i] = 0
            elif 67.5<=theta[j,i]<112.5:
                if mask[1,0]>D[j,i] or mask[1,2]>D[j,i]:
                    nms[j,i] = 0
            elif 112.5<=theta[j,i]<157.5:
                    if mask[0,0]>D[j,i] or mask[2,2]>D[j,i]:
                        nms[j,i] = 0
    
    
    
    
    
    ## Hysteresis thresholding    
    F = nms*(D>=t_high)
    coords = np.where((nms-F)>0)
    
    for k in range(0,int(np.sum(nms-F))):
        j = coords[0][k]
        i = coords[1][k]
        nms2 = F[j-1:j+2,i-1:i+2]
        nms2[1,1] = 0
        if D[j,i]>= t_low and np.sum(nms2)>0:
            F[j,i] = 1


    return F   
    
    
    
    
    
    
