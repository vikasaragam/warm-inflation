import numpy as np
# from numpy import array as arr
from scipy.integrate import odeint, cumtrapz
import matplotlib.pyplot as plt

Mp = 1
gstar = 228.75

Qmin = 10
Qmax = 150
numQ = 6
Qrange = np.linspace(Qmin, Qmax, numQ)
Qval = Qrange[0]

fval = Mp
mvalexp = -2
mval = 10**mvalexp * Mp

#%% Define functions

def V(phi, f, m):
    pot = (m*f)**2 * (1 + np.cos(phi/f))
    return pot

def dVdphi(phi, f, m):
    return -m**2 * f * np.sin(phi/f)

def H(phi, phid, rho, f, m):
    hubble = np.sqrt( 1/(3*Mp**2) * ( V(phi, f, m) + 1/2 * phid**2 + rho ) )
    return hubble

def backgroundEoms(IC, t, f, m, Q):
    phi, phid, rho = IC
    phidd = -3*H(phi, phid, rho, f, m)*(1+Q)*phid - dVdphi(phi, f, m)
    rhod = -4*H(phi, phid, rho, f, m)*rho + 3*H(phi, phid, rho, f, m)*Q*phid**2
    return [phid, phidd, rhod]

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

#%% Solve background equations of motion

Ntime = 10**5 + 1
ti = 0
tf = 10**3
dt = (tf - ti) / Ntime
time = np.linspace(ti, tf, Ntime)

inf0 = 10**-3
infd0 = 10**-3
rad0 = Qval*infd0**2 / 4
ic = [inf0, infd0, rad0]

backgroundSoln = odeint(backgroundEoms, ic, time, args=(fval, mval, Qval))

#%% Extract solution
infsoln = backgroundSoln[:,0]
infdsoln = backgroundSoln[:,1]
radsoln = backgroundSoln[:,2]

#%% Compute dynamical quantities

Vsoln = V(infsoln, fval, mval)
Hsoln = H(infsoln, infdsoln, radsoln, fval, mval)
eHsoln = -np.gradient(Hsoln, time) / (Hsoln**2)
etaHsoln = np.gradient(eHsoln, time) / (Hsoln * eHsoln)

#%% Compute end of inflation

indEnd = find_nearest(eHsoln, 1)
tend = time[indEnd]

#%% Compute efolds and CMB horizon crossing time

timeInf = time[:indEnd+1]
Ne = cumtrapz(Hsoln[:indEnd+1], timeInf, initial=0)
Nend = Ne[-1]

indHc = find_nearest(Ne, Nend-55)
timeHc = time[indHc]

#%% Plot inflaton solution

plt.xlabel('t')
plt.ylabel(r'$\phi$')
plt.plot(time, infsoln)

#%% Plot radiation energy density solution

plt.xlabel('t')
plt.ylabel(r'$\rho$')
plt.plot(time, radsoln)

#%% Plot potential

plt.xlabel('t')
plt.ylabel('V')
plt.plot(time, Vsoln)

#%% Plot H

plt.xlabel('t')
plt.ylabel('H')
plt.plot(time, Hsoln)

#%% Plot eH

plt.xlabel('t')
plt.ylabel(r'$\epsilon_H$')
plt.plot(time, eHsoln)

#%% Plot etaH

plt.xlabel('t')
plt.ylabel(r'$\eta_H$')
plt.plot(time, etaHsoln)

#%% Plot efolds

plt.xlabel('t')
plt.ylabel(r'$N_e$')
plt.plot(timeInf, Ne)