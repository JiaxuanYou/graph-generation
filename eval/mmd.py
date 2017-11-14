import networkx as nx
import numpy as np
from scipy.linalg import toeplitz
import pyemd

def gaussian_emd(x, y, sigma):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D histograms of two distributions with the same support
      sigma: standard deviation
    '''
    support_size = len(x)
    distance_mat = toeplitz(range(support_size)).astype(np.float)
    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))

def disc(samples1, samples2, kernel, *args, **kwargs):
    ''' Discrepancy between 2 samples
    '''
    d = 0
    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, *args, **kwargs)
    d /= len(samples1) * len(samples2)
    return d

def mmd(samples1, samples2, kernel, *args, **kwargs):
    ''' MMD between two samples
    '''
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
            disc(samples2, samples2, kernel, *args, **kwargs) - \
            2 * disc(samples1, samples2, kernel, *args, **kwargs)

def test():
    s1 = np.array([0.2, 0.8])
    s2 = np.array([0.3, 0.7])
    samples1 = [s1, s2]
    
    s3 = np.array([0.25, 0.75])
    s4 = np.array([0.35, 0.65])
    samples2 = [s3, s4]

    s5 = np.array([0.8, 0.2])
    s6 = np.array([0.7, 0.3])
    samples3 = [s5, s6]

    print('between samples1 and samples2: ', mmd(samples1, samples2, kernel=gaussian_emd, sigma=1))
    print('between samples1 and samples3: ', mmd(samples1, samples3, kernel=gaussian_emd, sigma=1))
    
if __name__ == '__main__':
    test()

