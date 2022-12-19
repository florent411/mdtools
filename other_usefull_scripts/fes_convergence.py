#!/usr/bin/env python 

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Q_name = argv[1] # current
P_name = argv[2] # ref
T = float(argv[3])

kB = 1.3806503e-23 # J.K^-1
NA = 6.02214076e23 # mol^-1
kT = (kB * NA * T) / 1000. # kJ.mol^-1

def clean_array(arr):
    mask = arr != float("inf")
    return mask

Q_x = np.loadtxt(Q_name, usecols=0, comments="#")
Q_y = np.loadtxt(Q_name, usecols=1, comments="#")
P_x = np.loadtxt(P_name, usecols=0, comments="#")
P_y = np.loadtxt(P_name, usecols=1, comments="#")

print(Q_x, Q_y, P_x, P_y)

# remove inf points
Q_x_clean = Q_x[clean_array(Q_y)]
Q_y_clean = Q_y[clean_array(Q_y)]
P_x_clean = P_x[clean_array(P_y)]
P_y_clean = P_y[clean_array(P_y)]

# interpolate 
Q_y_interp = np.interp(P_x_clean, Q_x_clean, Q_y_clean)

# calc. probabilities & norm. probabilities
Q = np.exp(-Q_y_interp / kT)
P = np.exp(-P_y_clean / kT)
Q_norm = Q / np.sum(Q)
P_norm = P / np.sum(P)

# Kullback-Leibler divergence
KLdiv = np.sum(P_norm * np.log(P_norm / Q_norm))

# Alonso & Echenique metric
Q_var = Q_y_interp.var()
P_var = P_y_clean.var()
r = np.corrcoef(Q_y_interp, P_y_clean)[0,1]
dA = np.sqrt((Q_var + P_var) * (1 - r**2))

print(KLdiv, dA)


# deltaFES
# # number of free-energy profiles
# nfes=$(( $1 -1 ))
# # minimum of basin A
# minA=$2
# # maximum of basin A
# maxA=$3
# # minimum of basin B
# minB=$4
# # maximum of basin B
# maxB=$5
# # temperature in energy units
# kbt=$6

# for i in `seq 0 ${nfes}`
# do
#  # calculate free-energy of basin A
#  A=`awk 'BEGIN{tot=0.0}{if($1!="#!" && $1>min && $1<max)tot+=exp(-$2/kbt)}END{print -kbt*log(tot)}' min=${minA} max=${maxA} kbt=${kbt} fes_${i}.dat`
#  # and basin B
#  B=`awk 'BEGIN{tot=0.0}{if($1!="#!" && $1>min && $1<max)tot+=exp(-$2/kbt)}END{print -kbt*log(tot)}' min=${minB} max=${maxB} kbt=${kbt} fes_${i}.dat`
#  # calculate difference
#  Delta=`echo "${A} - ${B}" | bc -l`
#  # print it
#  echo $i $Delta
# done
