 -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 01:56:31 2021

@author: Ayushi
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import numpy as np
from SAIRD import Model
import matplotlib.pyplot as plt


# Total population, N.
N = float(1360000000)

# Initial number of infected and recovered individuals, I0 and R0.
A0, I0, RA0,RI0, D0 = 20000, 13742, 0, 0 ,104

SA0= (0.2 * N)
#CONFUSION
# Everyone else, S0, is susceptible to infection initially.
SI0 = N - I0 - RA0 -RI0- A0 - SA0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days). 


# A grid of time points (in days)
# The SAIRD model differential equations.
def deriv(y, t,N,beta, gamma, delta):
    SA, SI, A, I, RA, RI,D = y
    dSAdt = -(beta * SA * A + beta * SA * I)/N 
    dSIdt = -(beta * SI * A + beta * SI * I)/N
    dAdt = ((beta * SA * A + beta * SA * I)/N)- gamma * A 
    dIdt = ((beta * SI * A + beta * SI * I)/N)- gamma * I 
    dRAdt = gamma * A 
    dRIdt = (gamma * I) - (delta *I)
    dDdt = delta * I
    return dSAdt,dSIdt, dAdt, dIdt, dRAdt ,dRIdt , dDdt


#CONFUSION
# Initial conditions vector
def Model(t,beta,gamma,delta):
    y0 = SA0 , SI0, A0, I0, RA0 ,RI0 , D0
    # Integrate the SIR equations over the time grid, t.
    return integrate.odeint(deriv, y0, t, args=(N, beta, gamma, delta))[:,3]
covid_data = pd.read_csv("E:\\Summer Project\\India_covid-19_data2.csv",parse_dates=["Date"])
Total_cases = covid_data[covid_data["Location"] == "India"]["New_cases"].values
days = len(Total_cases)
xdata = np.linspace(0, days - 1, days, dtype=float)
ydata = Total_cases
ydata = np.array(ydata, dtype=float)
fitted = Model(xdata, 0.234,0.06,0.008)
plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted/1000)
plt.title("Fit of SIR model for India infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()
