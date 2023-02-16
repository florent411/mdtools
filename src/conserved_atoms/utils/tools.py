#!/usr/bin/env python 

''' General tools for opes_postprocessing '''

import os
import sys
import numpy as np

# Function to calculate the Mahalanobis distance
# https://www.machinelearningplus.com/statistics/mahalanobis-distance/
def calculate_mahalanobis(y=None, data=None, cov=None):
  
    y_mu = y - data.mean()
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)

    return mahal.diagonal()
