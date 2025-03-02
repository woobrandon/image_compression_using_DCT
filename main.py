import numpy as np
from PIL import Image
from numpy import asarray
import math

def importim(im):
    image = Image.open(im)
    return asarray(image)

def rowcolumndecomp(patch,lambdak):
    (N,M) = np.shape(patch)
    cmatrix = np.zeros(shape = (N,M))
    for i in range(M):
        cmatrix[:,i] = DCT1d(patch[:,i],lambdak)
    for i in range(N):
        cmatrix[i,] = DCT1d(cmatrix[i,],lambdak)
    return cmatrix

def DCT1d(vector,lambdak):
    M = np.shape(vector)[0]
    cmatrix = np.zeros(shape = (M))
    a = ((2 / M) ** 0.5)
    b = math.pi / M
    for n in range(M):
        temp = 0
        c = n * b
        for k in range(M):
            temp += vector[k] * math.cos(c * (k + 0.5))
        cmatrix[n] = lambdak[n] * temp * a
    return cmatrix

def dct2d(im):
    (n,m) = np.shape(im)
    lambdak = [1/(2**0.5)] + [1]*7
    cmatrix = np.zeros(shape=(n,m))
    for i in range (n//8):
        for j in range(m//8):
            cmatrix[i*8:(i*8)+8,j*8:(j*8)+8] = rowcolumndecomp(im[i*8:(i*8)+8,j*8:(j*8)+8],lambdak)
    return cmatrix

def idct2d(coef):
    (n,m) = np.shape(coef)
    lambdak = [1/(2**0.5)] + [1]*7
    output = np.zeros(shape = (n,m))
    for i in range (n//8):
        for j in range(m//8):
            output[i*8:(i*8)+8,j*8:(j*8)+8] = inverserowcolumn(coef[i*8:(i*8)+8,j*8:(j*8)+8],lambdak)
    return output

def inverserowcolumn(patch, lambdak):
    (N,M) = np.shape(patch)
    qmatrix = np.zeros(shape = (N,M))  
    for i in range(M):
        qmatrix[:,i] = iDCT1d(patch[:,i],lambdak)
    for i in range(N):
        qmatrix[i,] = iDCT1d(qmatrix[i,],lambdak)
    return qmatrix

def iDCT1d(vector, lambdak):
    N = np.shape(vector)[0]
    qmatrix = np.zeros(shape = (N))
    a = (2 / N) ** 0.5
    b = math.pi / N
    for k in range(N):
        temp = 0
        c = k + 0.5
        for n in range(N):
            temp += lambdak[n] * vector[n] * math.cos(n * b * c)
        qmatrix[k] = round(a * temp)
    return qmatrix

def imcompress(im,r):
    (N,M) = np.shape(im)
    numpixels = round(N*M/r)
    output = np.zeros(shape = (N,M))
    flattenarray = im.flatten()
    flattenarray = [-abs(x) for x in flattenarray]
    position = {}
    for i in range(len(flattenarray)):
        if flattenarray[i] in position.keys():
            position[flattenarray[i]].append(i)
        else:
            position[flattenarray[i]] = [i]
    position = (dict(sorted(position.items()))).values()
    positionlist = []
    counter = 0
    for lis in position:
        for ele in lis:
            a = ele // N
            b = ele % N
            output[a,b] = im[a,b]
            counter += 1
            if counter == numpixels:
                return output

def imdecomp(coef):
    return idct2d(coef)

def showim(im):
    return (Image.fromarray(im)).show()