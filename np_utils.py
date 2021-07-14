import numpy as np



def min_max_normalize(ndarray):
  nmax, nmin = ndarray.max(), ndarray.min()
  normalized = (ndarray - nmin) / (nmax - nmin)
  
  return normalized