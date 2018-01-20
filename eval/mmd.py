import concurrent.futures
from functools import partial
import networkx as nx
import numpy as np
from scipy.linalg import toeplitz
import pyemd

def emd(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd

def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))

def gaussian(x, y, sigma=1.0):
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))

def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d

def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples
    '''
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker, 
                    [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                d += dist
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' MMD between two samples
    '''
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
            disc(samples2, samples2, kernel, *args, **kwargs) - \
            2 * disc(samples1, samples2, kernel, *args, **kwargs)

def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' EMD between average of two samples
    '''
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples2, kernel, *args, **kwargs),[samples1[0],samples2[0]]


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

    print('between samples1 and samples2: ', compute_mmd(samples1, samples2, kernel=gaussian_emd,
        is_parallel=False, sigma=1.0))
    print('between samples1 and samples3: ', compute_mmd(samples1, samples3, kernel=gaussian_emd,
        is_parallel=False, sigma=1.0))
    
if __name__ == '__main__':
    test()

