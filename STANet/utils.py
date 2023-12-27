import os
import pickle
import numpy as np
import struct

def loadYUVfile(filename, width, height, idxFrames, colorSampling, bitDepth):

    numFrames = idxFrames.size

    if bitDepth == 8:
        multiplier = 1
        elementType = 'B'
    elif bitDepth == 10:
        multiplier = 2
        elementType = 'H'
    else:
        raise Exception("Error reading file: bit depth not allowed (8 or 10 )")

    if colorSampling == 420:
        sizeFrame = 1.5 * width * height * multiplier
        width_size = int(width / 2)
        height_size = int(height / 2)
    elif colorSampling == 422:
        sizeFrame = 2 * width * height * multiplier
        width_size = int(width / 2)
        height_size = height
    elif colorSampling == 444:
        sizeFrame = 3 * width * height * multiplier
        width_size = width
        height_size = height
    else:
        raise Exception("Error reading file: color sampling not allowed (420, 422 or 444 )")

    sizeY = width * height
    sizeColor = width_size * height_size

    with open(filename,'rb') as fileID:

        fileID.seek(int(idxFrames[0]*sizeFrame),0)

        for iframe in range (0,numFrames):

            try:
                buf = struct.unpack(str(sizeY)+elementType, fileID.read(sizeY*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufY = np.reshape(buf, (height,width))

            try:
                buf = struct.unpack(str(sizeColor)+elementType, fileID.read(sizeColor*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufU = np.reshape(buf, (height_size,width_size))

            try:
                buf = struct.unpack(str(sizeColor)+elementType, fileID.read(sizeColor*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufV = np.reshape(buf, (height_size,width_size))

            if colorSampling == 420:
                bufU = bufU.repeat(2, axis=0).repeat(2, axis=1)
                bufV = bufV.repeat(2, axis=0).repeat(2, axis=1)
            elif colorSampling == 422:
                bufU = bufU.repeat(2, axis=1)
                bufV = bufV.repeat(2, axis=1)

            image = np.stack((bufY,bufU,bufV), axis=2)
            image.resize((1,height,width,3))

            if iframe == 0:
                video = image
            else:
                video = np.concatenate((video,image), axis=0)

    return video


def yuv2rgb(image, bitDepth):

    N = ((2**bitDepth)-1)

    Y = np.float32(image[:,:,0])
    
    U = np.float32(image[:,:,1])
    
    V = np.float32(image[:,:,2])

    Y = Y/N
    U = U/N
    V = V/N

    fy = Y
    fu = U-0.5
    fv = V-0.5

    # parameters
    KR = 0.2627
    KG = 0.6780
    KB = 0.0593 

    R = fy + 1.4746*fv
    B = fy + 1.8814*fu
    G = -(B*KB+KR*R-Y)/KG

    R[R<0] = 0
    R[R>1] = 1
    G[G<0] = 0
    G[G>1] = 1
    B[B<0] = 0
    B[B>1] = 1

    rgb_image = np.array([R,G,B])
    rgb_image = np.swapaxes(rgb_image,0,2)
    rgb_image = np.swapaxes(rgb_image,0,1)
    rgb_image = rgb_image*N

    return rgb_image
