import numpy as np 
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import lmfit
import numpy as np
from SAIRD import Model
from scipy import integrate, optimize
import matplotlib.pyplot as plt


# Total population, N.
N = float(1360000000)

# Initial number of infected and recovered individuals, I0 and R0.
A0, I0, RA0,RI0, D0 = 0, 1, 0, 0 ,0

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
 

covid_data = pd.read_csv("E:\\Summer Project\\India_covid-19_data3.csv",parse_dates=["Date"])
Total_cases = covid_data[covid_data["Location"] == "India"]["New cases"].values
days = len(Total_cases)
xdata = np.linspace(0, days - 1, days, dtype=float)
ydata = Total_cases
ydata = np.array(ydata, dtype=float)
fitted = Model(xdata, 0.12,0.02,0.005)
plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted/6000)
plt.title("Fit of SAIRD model for India infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()
