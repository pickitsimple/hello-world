import Odderon
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

test = Odderon.Chisq("pp-1.96TeV-D0.csv")

m = Minuit(test.ChisqCalpp, M0=100, gOV=0.8, gPNN=2.5, aOp=0.2, eO=0.05, A=1, B=1, L=1)

m.migrad()  # run optimiser
m.hesse()   # run covariance estimator

# print(m.values)
# print(m.errors)

M0min, gOVmin, gPNNmin, aOpmin, eOmin, Amin, Bmin, Lmin = m.values
M0err, gOVerr, gPNNerr, aOperr, eOerr, Aerr, Berr, Lerr = m.errors

# chimin = test.ChisqCalpp(M0min, gOVmin, gPNNmin, aOpmin, eOmin, Amin, Bmin, Lmin)
# print(chimin)


def mockfunc(x, x0, sigma):
	return (x-x0)**2/sigma**2

taglist = ['M0','gOV','gPNN','aOp','eO','A','B','L']

for t in taglist:
	x0, sigma, x, y = test.ErrEst(m.values,m.errors,test.ChisqCalpp,tag=t)
	print(x0,sigma)
	dent=80
	xplot = x[dent:-dent]
	yplot = y[dent:-dent]
	plt.plot(xplot,yplot,label=r'$\chi^2$')
	plt.plot(xplot,mockfunc(xplot, x0, sigma),label=r'$\frac{(x-x_0)^2}{\sigma^2}$')
	plt.title("{} = {:.4f} + {:.4f}".format(t,x0,sigma))
	plt.xlabel('{}'.format(t))
	plt.ylabel(r'$\chi^2$')
	plt.legend()
	plt.savefig('figs/{}_sigma.jpg'.format(t))
	plt.show()