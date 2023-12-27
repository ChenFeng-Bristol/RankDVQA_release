from .lpips_3d import *


def get_model(modelname, depth_ksize=1, opt=None):
    if modelname == 'multiscale_v33':
        moduleNetwork = LPIPS_3D_Diff(net='multiscale_v33').cuda()
    else:
        raise NotImplementedError
    return moduleNetwork
