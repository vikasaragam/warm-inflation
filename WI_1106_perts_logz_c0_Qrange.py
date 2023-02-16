import numpy as np
from numpy import array as arr
from numpy.random import normal
# from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%% Define parameters

# expi = 3
# expf = 0#-1
# zi = 10**expi
# zf = 10**expf

qi = 6#np.log(zi)# 6 to match Luca
qf = 0#np.log(zf)# -4 to match Luca

N = 10**5# 10**5 to match Luca
dq = (qf - qi)/N
qtime = np.linspace(qi, qf, N)
# dz = (zf - zi)/N
# ztime = np.linspace(zi, zf, N)
# ztime = np.logspace(expi, expf, N)# Doesn't seem to give great results :\

#%% Define thermal noise and equations of motion

xiMean = 0
# xiVar = 1
# np.sqrt(deltaz)

# def xi(z, mean, stdev):
#     noise = normal(mean, stdev)
#     return noise

# xi = normal(xiMean, xiVar, N)

def eoms(ini, q, c, Q):
    y, vy, w, u = ini
    z = np.exp(q)
    dvydq = 3*(1+Q) * vy - z**2 * y - 3*c*Q * w# + xi(z, xiMean, xiVar)
    dwdq = (4-c) * w - z**2/(3*Q) * u + 2*vy
    dudq = 3 * u + Q * (w + 3*y)
    
    # dvwdz = (8-c)/z * vw - ( (20-5*c+6*c*Q)/z**2 + 1/3 )*w + (6*Q-4)/z * vy - 3*y# + 2*xi(z, xiMean, xiVar)
    return [vy, dvydq, dwdq, dudq]

def solver(IC, q, c, Q):
    y2avg = np.zeros(N)
    w2avg = np.zeros(N)
    u2avg = np.zeros(N)
    
    for j in range(Nruns):
        ysoln = np.zeros(N)#[None for p in range(N)]
        vysoln = np.zeros(N)#[None for p in range(N)]
        wsoln = np.zeros(N)#[None for p in range(N)]
        usoln = np.zeros(N)#[None for p in range(N)]
        
        ysoln[0], vysoln[0], wsoln[0], usoln[0] = IC
        
        newIC = IC
        
        for i in range(N-1):
            # dz = ztime[i+1] - ztime[i]# Compute dz each time step in case we are using logspaced time. If not, just compute this once outside the loop
            derivs = eoms(newIC, q[i], c, Q)
            
            dy = derivs[0] * dq
            dvy = derivs[1] * dq
            dw = derivs[2] * dq
            du = derivs[3] * dq
            
            ysoln[i+1] = ysoln[i] + dy
            vysoln[i+1] = vysoln[i] + dvy + (np.exp(q[i]))**1.5 * normal(xiMean, np.sqrt(-dq))
            wsoln[i+1] = wsoln[i] + dw
            usoln[i+1] = usoln[i] + du
            
            newIC = [ ysoln[i+1], vysoln[i+1], wsoln[i+1], usoln[i+1] ]
        
        # y2solnarr = arr(ysoln)**2
        # w2solnarr = arr(wsoln)**2
        # u2solnarr = arr(usoln)**2
        
        # y2avg = y2avg + y2solnarr/Nruns
        # w2avg = w2avg + w2solnarr/Nruns
        # u2avg = u2avg + u2solnarr/Nruns
        
        y2avg = y2avg + ysoln**2 / Nruns
        w2avg = w2avg + wsoln**2 / Nruns
        u2avg = u2avg + usoln**2 / Nruns
    
    return [y2avg, w2avg, u2avg]


#%% Solve perturbations' equations of motion for range of Q values using Euler-Maruyama method

Nruns = 1000

cval = 0

Qmin = 10
Qmax = 150
# deltaQ = 28
numQ = 6#int((Qmax - Qmin)/deltaQ + 1)
Qval = np.linspace(Qmin, Qmax, numQ)

ywuIC = [1,0,0,0]# y(0) = 1, y'(0) = w(0) = u(0) = 0

y2horizon = np.zeros(numQ)
w2horizon = np.zeros(numQ)
u2horizon = np.zeros(numQ)

for k in range(numQ):
    y2, w2, u2 = solver(ywuIC, qtime, cval, Qval[k])
    
    y2horizon[k] = y2[-1]
    w2horizon[k] = w2[-1]
    u2horizon[k] = u2[-1]
    
    print("Q = " + str(Qval[k]) + " done")


#%% Plot horizon crossing values vs Q

def y2c0(Q):
    return np.sqrt(3*np.pi)/4 * np.sqrt(1+Q)/(1+3*Q)

def cgtr0fit(Q, alpha, beta, A, B):
    return A*Q**alpha + B*Q**beta

plt.xlabel("Q")
plt.ylabel("$\\langle\\hspace{0.2} y^2 \\hspace{0.1}\\rangle_*$")
plt.yscale("log")
plt.plot(Qval, y2horizon, '.', Qval, y2c0(Qval) )

#%% Plot solution

plt.xscale("log")
plt.xlabel("z = k/(aH)")
plt.plot(np.exp(qtime), y2)

#%%
plt.plot(np.exp(qtime[0:int(N/100)]), y2[0:int(N/100)])

#%% Store horizon crossing values

dataHorizon = arr([Qval, y2horizon, w2horizon, u2horizon])
dataHorizon = dataHorizon.T

path  = '/Users/vikasaragam/Documents/Python/'
int_file = path + 'results_c0_Q' + str(Qmin) + '-' + str(Qmax) + '_' + str(Nruns) + 'runs.txt'
np.savetxt(int_file, dataHorizon)

#%% Store data for one Q value

data = arr([ztime, y2, w2, u2])
data = data.T

path  = '/Users/vikasaragam/Documents/Python/'
int_file = path + 'results_c0_Q100.txt'
np.savetxt(int_file, data)

#%% Import horizon crossing data

path  = '/Users/vikasaragam/Documents/Python/'
file = 'results_c0_Q10-150_1000runs.txt'#'results_c1_Q10-3000_1000runs.txt'

Qval, y2horizon, w2horizon, u2horizon = np.loadtxt(path+file, unpack=True)

#%% Import background horizon crossing data

path  = '/Users/vikasaragam/Documents/Python/'
path_file = path + 'results_backgroundHc_constantQ_index_time_H_phidot_rhorad.txt'

indHc, timeHc, Hhc, infdHc, radHc = np.loadtxt(path_file, unpack=True)

#%% Compute curvature scalar power spectrum

Mp = 1
gstar = 228.75

TempHc = (30/(gstar * np.pi**2) * radHc)**(1/4)

powerR = (Hhc/infdHc)**2 * Hhc*(1 + 3*Qval)*TempHc/(np.pi**2) * y2horizon

#%% Compute unscaled curvature scalar power spectrum and G(Q)

powerR_unscaled = (Hhc**2/(2*np.pi*infdHc))**2 * ( 1 + 2/(np.exp(Hhc/TempHc)-1) + 2*np.sqrt(3)*np.pi*Qval/np.sqrt(3 + 4*np.pi*Qval) * TempHc/Hhc )
GofQ = powerR / powerR_unscaled

#%% Compute fit for G(Q)

def GofQfit(x, a, b, c, d):
    return 1 + a*x**b + c*x**d

popt, pcov = curve_fit(GofQfit, Qval, GofQ)

#%% Plot horizon crossing power spectrum vs Q

plt.xlabel("Q")
plt.ylabel("$\\Delta^2_\\mathcal{R} $")
plt.yscale("log")
plt.plot(Qval, powerR, '.', Qval, powerR_unscaled, '.')#, Qval, powerR_unscaled*GofQfit(Qval, *popt))