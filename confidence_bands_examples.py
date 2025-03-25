import numpy as np
from scipy.special import xlogy
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib import rc


# Constants
__EPS__ = 1e-15

def KL_bin(q, p):
    return xlogy(q, q / p) + xlogy(1 - q, (1 - q) / (1 - p))

def Newton_KL(q, c, p0, iter):
    p = p0
    for i in range(iter):
        h_value = KL_bin(q, p) - c
        hp_value = (1 - q) / (1 - p) - q / p
        p = np.clip(p - h_value / np.clip(hp_value, a_min=__EPS__, a_max=None), a_min=None, a_max=1 - __EPS__)
    return p

def inv_KL(q, c, iter=50, **kwargs):  #Compute inverse of binary kl
    b = np.clip(q + np.sqrt(c / 2), None, 1 - __EPS__)
    return Newton_KL(q, c, b, iter)

def inverse_transform_sampling(cdf, size=1, x_min=0.0, x_max=1.0, num_points=1000): #Sample from cdf
    x = np.linspace(x_min, x_max, num_points)
    y = cdf(x)
    inverse_cdf = interp1d(y, x, kind='linear', fill_value='extrapolate')
    uniform_samples = np.random.uniform(0, 1, size)
    samples = inverse_cdf(uniform_samples)
    return samples

def cdf1(x):
    return x

def cdf2(x):
    return np.sin(np.sqrt(x)*np.pi/2)**6

def cdf3(x):
    return (3/4 + 1/4*(x>.6))*x + .1*np.sin(2*np.pi*x**.9)**3

Y = np.linspace(0, 1, 1000)
CDF1 = cdf1(Y)
CDF2 = cdf2(Y)
CDF3 = cdf3(Y)

def ecdf(samples, y): #gives the empirical cdf function given a dataset (fraction of data points upper bounded by y)
    return np.sum(samples[:, np.newaxis]<=y, axis=0)/len(samples)

def ub(Y, ssamples, delta):
  T = len(ssamples)
  out = []
  for y in Y:
    CT = np.searchsorted(ssamples, y, side='right') #number of points in sample that are smaller or equal to y
    tt = np.arange(CT, T) #values of t to be tested (T is omitted)
    Xttp = ssamples[tt] #these are shifted, as we need X_{t+1} if indexing starts from 1
    try: out.append(min(inv_KL(tt/T, np.log(2*np.sqrt(T)/((Xttp-y)*delta))/T)))
    except: out.append(1) #this is what happens if T had to be picked...
  return np.array(out)

def lb(Y, ssamples, delta):
  T = len(ssamples)
  out = []
  for y in Y:
    CT = np.searchsorted(ssamples, y, side='right') #number of points in sample that are smaller or equal to y
    tt = np.arange(1, CT+1) #values of t to be tested (0 is omitted)
    Xtt = ssamples[tt-1]
    try: out.append(1-min(inv_KL(1-tt/T, np.log(2*np.sqrt(T)/((y-Xtt)*delta))/T)))
    except: out.append(0) #this is what happens if 0 had to be picked...
  return np.array(out)

T = 100
S1 = inverse_transform_sampling(cdf1, size=T)
S2 = inverse_transform_sampling(cdf2, size=T)
S3 = inverse_transform_sampling(cdf3, size=T)
delta = .05

sortS1 = np.sort(S1)
sortS2 = np.sort(S2)
sortS3 = np.sort(S3)

LB1 = lb(Y, sortS1, delta)
UB1 = ub(Y, sortS1, delta)
LB2 = lb(Y, sortS2, delta)
UB2 = ub(Y, sortS2, delta)
LB3 = lb(Y, sortS3, delta)
UB3 = ub(Y, sortS3, delta)

rc('text', usetex=True)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

ax1.plot(Y, LB1, c='salmon')
ax1.plot(Y, UB1, c='salmon')
ax1.fill_between(Y, UB1, LB1, alpha=.3, color='salmon')
ax1.plot(Y, cdf1(Y), c='black')
ax1.set_xlabel(r'$y$', fontsize=14)
ax1.set_ylabel(r'$F$', labelpad=15, rotation=0, fontsize=14)

ax2.plot(Y, LB2, c='salmon')
ax2.plot(Y, UB2, c='salmon')
ax2.fill_between(Y, UB2, LB2, alpha=.3, color='salmon')
ax2.plot(Y, cdf2(Y), c='black')
ax2.set_xlabel(r'$y$', fontsize=14)

ax3.plot(Y, LB3, c='salmon')
ax3.plot(Y, UB3, c='salmon')
ax3.fill_between(Y, UB3, LB3, alpha=.3, color='salmon')
ax3.plot(Y[Y<.6], cdf3(Y[Y<.6]), c='black')
ax3.plot(Y[Y>.6], cdf3(Y[Y>.6]), c='black')
ax3.set_xlabel(r'$y$', fontsize=14)

plt.tight_layout()
plt.show()
