# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:37:51 2021

@author: Ayushi
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1360000000
# Initial number of infected and recovered individuals, I0 and R0.
A0, I0, R0 = 0, 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - A0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma, delta = 0.2, 0.08, 0.001 
# A grid of time points (in days)
t = np.linspace(0, 500, 500)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, delta):
    S, A, I, R = y
    dSdt = -(beta * S * A + beta * S * I)/N
    dAdt = ((beta * S * A + beta * S * I)/N)- gamma * A - delta * A
    dIdt = delta * A - gamma * I
    dRdt = gamma * (I + A)
    return dSdt, dAdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, A0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
S, A, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, A/1000, 'y', alpha=0.5, lw=2, label='Asymptomatic')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
