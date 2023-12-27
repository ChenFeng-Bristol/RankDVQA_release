import os
import pickle
from networks.lpips_3d import *
import torchvision.transforms as transforms
import cv2
from PIL import Image
import struct
from scipy.stats import spearmanr, pearsonr,kendalltau
from pathlib import Path
from scipy.optimize import curve_fit

def modelFun(b, x):
    return (b[0]-b[1]) / (1 + np.exp((b[2]-x) / abs(b[3]))) + b[1]

def compute_metrics(x, y, STD=None):
    if STD is not None:
        # If BVIHD_STD is provided, use it for weighted regression
        ww = 1.0 / STD
        yw = ww * y
    else:
        # Otherwise, all weights are set to 1 (equivalent to non-weighted regression)
        ww = np.ones_like(y)
        yw = y
        
    # Initial guess for parameters
    bStart = [np.max(y), np.min(y), np.mean(x), 1]
    
    # Weighted non-linear fitting
    popt, _ = curve_fit(lambda x, b0, b1, b2, b3: ww * modelFun([b0, b1, b2, b3], x), x, yw, p0=bStart, maxfev=10000)
    
    # Compute predictions using the model
    yhat = modelFun(popt, x)
    
    # Compute metrics
    SROCC, _ = spearmanr(x, y)
    PLCC, _ = pearsonr(yhat, y)
    RMSE = np.sqrt(np.mean((yhat - y) ** 2))
    OR = np.nan
    if STD is not None:
        OR = np.sum(np.abs(yhat - y) > 2.0 / ww) / len(x)
    
    return SROCC, PLCC, RMSE, OR


def read_yuv_frame(fid, width, height, bitDepth):
    if bitDepth == 8:
        dtype = np.uint8
    elif bitDepth == 10:
        dtype = np.uint16
    else:
        raise Exception("Error reading file: bit depth not allowed (8 or 10 )")


    y_size = width * height
    y_data = np.frombuffer(fid.read(y_size * dtype().itemsize), dtype).reshape((height, width))
    
    uv_size = y_size // 2  
    fid.read(uv_size * dtype().itemsize)
    
    if bitDepth == 10:
        y_data = y_data / 1023.0
    if bitDepth == 8:
        y_data = y_data / 255.0
    
    return y_data.astype(np.float32)


def mkdirifnotexist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dirpath(path, level=1):
    dirpath = path
    for i in range(level):
        dirpath = os.path.dirname(dirpath)
    return dirpath



def get_dataset_dir(dataset='rankdvqa'):
    if dataset == 'rankdvqa':
        datadir = '/user/work/cf18202/VMAF_RANKing/'
    elif dataset == 'bvivfi':
        datadir = 'BVI-VFI_DATASET_PATH'
    else:
        raise ValueError
    return datadir


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def writelist(filepath, src):
    with open(filepath, 'w') as f:
        for item in src:
            f.write('%s\n'%item)


def readlist(filepath):
    with open(filepath) as f:
        rst = f.read().splitlines()
    return rst


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def yuv_to_rgb_single_frame(yuv_file, width, height, bit_depth=10, target_frame=0):
    
    if bit_depth == 8:
        dtype = np.uint8
        frame_size = width * height + (width // 2 * height // 2) * 2
    elif bit_depth == 10:
        dtype = np.uint16
        frame_size = 2 * (width * height + (width // 2 * height // 2) * 2)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    with open(yuv_file, 'rb') as f:

        f.seek(target_frame * frame_size)

        y = np.frombuffer(f.read(width * height * dtype().itemsize), dtype=dtype).reshape((height, width))
        u = np.frombuffer(f.read(width * height // 4 * dtype().itemsize), dtype=dtype).reshape((height // 2, width // 2))
        v = np.frombuffer(f.read(width * height // 4 * dtype().itemsize), dtype=dtype).reshape((height // 2, width // 2))

        if bit_depth == 10:
            y = (y >> 2).astype(np.uint8)
            u = (u >> 2).astype(np.uint8)
            v = (v >> 2).astype(np.uint8)

        y = cv2.resize(y, (width, height), interpolation=cv2.INTER_LINEAR)
        u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

        yuv_image = cv2.merge([y, u, v])
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        
        return rgb_image
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
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
    rgb_image = rgb_image*255

    return rgb_image
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