import numpy as np
from scipy.special import xlogy
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from matplotlib import rc
from matplotlib.ticker import FuncFormatter


# Constants
__EPS__ = 1e-15

def KL_bin(q, p): #compute binary kl
    return xlogy(q, q / p) + xlogy(1 - q, (1 - q) / (1 - p))

def Newton_KL(q, c, p0, iter): #Newton's method to invert kl
    p = p0
    for i in range(iter):
        h_value = KL_bin(q, p) - c
        hp_value = (1 - q) / (1 - p) - q / p
        p = np.clip(p - h_value / np.clip(hp_value, a_min=__EPS__, a_max=None), a_min=None, a_max=1 - __EPS__)
    return p

def inv_KL(q, c, iter=50, **kwargs):  #Compute inverse of binary kl
    b = np.clip(q + np.sqrt(c / 2), None, 1 - __EPS__)
    return Newton_KL(q, c, b, iter)

def inverse_transform_sampling(cdf, size=1, x_min=0.0, x_max=1.0, num_points=1000): #Sample from given cdf
    x = np.linspace(x_min, x_max, num_points)
    y = cdf(x)
    inverse_cdf = interp1d(y, x, kind='linear', fill_value='extrapolate')
    uniform_samples = np.random.uniform(0, 1, size)
    samples = inverse_cdf(uniform_samples)
    return samples
  
def cdf(x): #cdf function used for experiment
    return np.sin(np.sqrt(x)*np.pi/2)**6

Y = np.linspace(0, 1, 500)
CDF = cdf(Y)

def ecdf(samples, y): #gives the empirical cdf function given a dataset (fraction of data points upper bounded by y)
    return np.sum(samples[:, np.newaxis]<=y, axis=0)/len(samples)

T1 = 100
T2 = 1000
T3 = 10000

S = inverse_transform_sampling(cdf, size=T3)

S1 = np.sort(S[:T1])
S2 = np.sort(S[:T2])
S3 = np.sort(S[:T3])
delta = .05


#ours
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


LB1 = lb(Y, S1, delta)
UB1 = ub(Y, S1, delta)
LB2 = lb(Y, S2, delta)
UB2 = ub(Y, S2, delta)
LB3 = lb(Y, S3, delta)
UB3 = ub(Y, S3, delta)

#uniform Howard
def ubdkw(Y, samples, delta):
  T = len(samples)
  return ecdf(samples, Y) + 0.85*np.sqrt((np.log(np.log(np.exp(1)*T)) + 4/5*np.log(1612/delta))/(T))

def lbdkw(Y, samples, delta):
  T = len(samples)
  return ecdf(samples, Y) - 0.85*np.sqrt((np.log(np.log(np.exp(1)*T)) + 4/5*np.log(1612/delta))/(T))


LB1dkw = lbdkw(Y, S1, delta)
UB1dkw = ubdkw(Y, S1, delta)
LB2dkw = lbdkw(Y, S2, delta)
UB2dkw = ubdkw(Y, S2, delta)
LB3dkw = lbdkw(Y, S3, delta)
UB3dkw = ubdkw(Y, S3, delta)


#adaptive Howard
def logit(p): return np.log(p/(1-p))
def invlogit(l): return np.exp(l)/(1 + np.exp(l))
def R(p, n): return np.where(p >= 0.5, p, np.minimum(0.5, invlogit(logit(p) + np.sqrt(2.1/n))))
def L(p, n): return 1.4*np.log(np.log(2.1*n)) + 1.4*np.log(1+np.sqrt(n)*np.abs(logit(p)))+np.log(72/delta)
def B(p, n): return delta*np.sqrt(2.1*n*R(p,n)*(1-R(p,n))) + 1.5*np.sqrt(R(p,n)*(1-R(p,n))*n*L(p, n)) + 0.81*L(p,n)

F1 = np.linspace(1.5*np.min(LB1dkw-CDF), 1.5*np.max(UB1dkw-CDF), 1000)
FF1, YY1 = np.meshgrid(F1, Y)
ZL1 = FF1+cdf(YY1) + B(FF1+cdf(YY1),T1)/T1 - ecdf(S1, Y)[:, np.newaxis]
ZU1 = FF1+cdf(YY1) - B(1-FF1-cdf(YY1),T1)/T1 - ecdf(S1, Y)[:, np.newaxis]
F2 = np.linspace(1.5*np.min(LB2dkw-CDF), 1.5*np.max(UB2dkw-CDF), 1000)
FF2, YY2 = np.meshgrid(F2, Y)
ZL2 = FF2+cdf(YY2) + B(FF2+cdf(YY2),T2)/T2 - ecdf(S2, Y)[:, np.newaxis]
ZU2 = FF2+cdf(YY2) - B(1-FF2-cdf(YY2),T2)/T2 - ecdf(S2, Y)[:, np.newaxis]
F3 = np.linspace(1.5*np.min(LB3dkw-CDF), 1.5*np.max(UB3dkw-CDF), 1000)
FF3, YY3 = np.meshgrid(F3, Y)
ZL3 = FF3+cdf(YY3) + B(FF3+cdf(YY3),T3)/T3 - ecdf(S3, Y)[:, np.newaxis]
ZU3 = FF3+cdf(YY3) - B(1-FF3-cdf(YY3),T3)/T3 - ecdf(S3, Y)[:, np.newaxis]


#plots
rc('text', usetex=True)
plt.rcParams["font.family"] = "serif"  # Use LaTeX's default serif font

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4),) #sharey=True)

ax1.contour(YY1, FF1, ZL1, levels=[0], linestyles=':', colors='teal')
ax1.contour(YY1, FF1, ZU1, levels=[0], linestyles=':', colors='teal')
ax1.plot(Y, np.maximum(LB1dkw,0)-CDF,'--', c='royalblue')
ax1.plot(Y, np.minimum(UB1dkw,1)-CDF, '--', c='royalblue')
ax1.plot(Y, LB1-CDF, c='salmon')
ax1.plot(Y, UB1-CDF, c='salmon')
ax1.fill_between(Y, UB1-CDF, LB1-CDF, alpha=.3, color='salmon')
ax1.plot(Y, CDF-CDF, c='black')
ax1.set_xlabel(r'$y$', fontsize=14)
ax1.set_title(r'$T=100$')


ax2.contour(YY2, FF2, ZL2, levels=[0], linestyles=':', colors='teal')
ax2.contour(YY2, FF2, ZU2, levels=[0], linestyles=':', colors='teal')
ax2.plot(Y, np.maximum(LB2dkw,0)-CDF,'--', c='royalblue')
ax2.plot(Y, np.minimum(UB2dkw,1)-CDF, '--', c='royalblue')
ax2.plot(Y, LB2-CDF, c='salmon')
ax2.plot(Y, UB2-CDF, c='salmon')
ax2.fill_between(Y, UB2-CDF, LB2-CDF, alpha=.3, color='salmon')
ax2.plot(Y, CDF-CDF, c='black')
ax2.set_xlabel(r'$y$', fontsize=14)
ax2.set_title(r'$T=1000$')


ax3.contour(YY3, FF3, ZL3, levels=[0], linestyles=':', colors='teal')
ax3.contour(YY3, FF3, ZU3, levels=[0], linestyles=':', colors='teal')
ax3.plot(Y, np.maximum(LB3dkw,0)-CDF,'--', c='royalblue')
ax3.plot(Y, np.minimum(UB3dkw,1)-CDF, '--', c='royalblue')
ax3.plot(Y, LB3-CDF, c='salmon')
ax3.plot(Y, UB3-CDF, c='salmon')
ax3.fill_between(Y, UB3-CDF, LB3-CDF, alpha=.3, color='salmon')
ax3.plot(Y, CDF-CDF, c='black')
ax3.set_xlabel(r'$y$', fontsize=14)
ax3.set_title(r'$T=10000$')


plt.tight_layout()

legend_lines = [
    Line2D([], [], color='salmon', linestyle='-', label='Ours'),
    Line2D([], [], color='royalblue', linestyle='--', label='Non-adaptive (HR22, Thm 2)'),
    Line2D([], [], color='teal', linestyle=':', label='Adaptive (HR22, Thm 5)')
]

#Add legend below the plot
fig.legend(
    handles=legend_lines,
    loc='center',
    bbox_to_anchor=(.5, 0.055),
    ncol=3,  # Places legend items in one row
    prop={"size": 14}
)

#Adjust layout to make space for the legend
fig.subplots_adjust(bottom=.25)

plt.show()

