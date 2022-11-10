import numpy as np
import matplotlib.pyplot as plt

class Chisq(object):
	"""docstring for Chisq"""
	def __init__(self, arg):
		#super(Chisq, self).__init__()
		self.arg = arg
		self.data = np.genfromtxt(self.arg,delimiter=',',dtype=float)

	def dsdtpp(self,s,t,M0,gOV,gPNN,aOp,eO,A,B,L):
		return -(1/s**2)*3.42905*10**-6*gOV**2*gPNN**2*np.absolute((M0**2*(-0.25*1j*s)**(0.25*t+0.0808)*(1.26016-t)**2*(1.26016-A*t)**2*(s**5+(2.5*t-8.79844)*s**4+(2.23611*t**2-16.9126*t+31.8252)*s**3+(0.854167*t**3-10.7781*t**2+44.1252*t-59.0295)*s**2+(0.135417*t**4-2.54788*t**3+17.821*t**2-52.6725*t+55.9818)*s+0.00868056*t**5-0.180246*t**4+1.95413*t**3-10.1693*t**2+24.2621*t-21.6471)*(1j*s*aOp)**(t*aOp+eO))/(s**3*(t-0.879844)**2*(t-0.71)**4*(B*t-0.879844)**2*(L*t-0.71)**4)) + (3.17505*10**(-8)*gOV**4*M0**4*(s**6+s**5*(3*t-10.5581)+s**4*(3.41667*t**2-25.5155*t+46.7056)+s**3*(1.83333*t**3-22.2894*t**2+87.2181*t-110.794)+s**2*(0.467785*t**4-8.49294*t**3+54.7538*t**2-149.882*t+148.576)+s*(0.0511188*t**5-1.34081*t**4+13.1661*t**3-60.1479*t**2+129.533*t-106.709)+0.00241127*t**6-0.066192*t**5+0.959294*t**4-6.86155*t**3+25.0297*t**2-45.1397*t+32.09)*(1.26016-A*t)**4*(aOp**2*s**2)**(aOp*t+eO))/(s**6*(B*t-0.879844)**4*(L*t-0.71)**8)+1/((t-0.879844)**4*(t-0.71)**8)*2.31257*10**(-6)*gPNN**4*np.exp(-0.693147*t)*(1.26016-t)**4*(32*s**4+s**3*(64*t-225.24)+s**2*(42*t**2-323.783*t+644.072)+s*(10*t**3-133.736*t**2+594.528*t-871.82)+t**4-14.0775*t**3+116.119*t**2-397.768*t+469.828)*(s**2)**(0.25*t-1.9192)

	def ChisqCal(self,func):
		# print(self.data)
		tdata=self.data[:,0]
		dsdtdata=self.data[:,1]
		dsdtmodel=func(tdata)
		chisq=sum(((dsdtmodel-dsdtdata)/dsdtdata)**2)
		return chisq
		
	def ChisqCalpp(self,M0,gOV,gPNN,aOp,eO,A,B,L):
		s=1960**2
		tdata=self.data[:,0]
		dsdtdata=self.data[:,1]
		dsdtmodel=self.dsdtpp(s,-tdata,M0,gOV,gPNN,aOp,eO,A,B,L)
		# print(dsdtmodel)
		# print(dsdtdata)
		# print(dsdtmodel-dsdtdata)
		chisq=sum(((dsdtmodel-dsdtdata)/dsdtdata)**2)
		return chisq

	def ErrEst(self, values, errors, chifunc, tag='name'):
		""" tag is for choosing the parameter you want to estimate (M0,gOV,gPNN,aOp,eO,A,B,L)
		values and errors are coming from m.values, m.errors respectively """
		M0min, gOVmin, gPNNmin, aOpmin, eOmin, Amin, Bmin, Lmin = values
		M0err, gOVerr, gPNNerr, aOperr, eOerr, Aerr, Berr, Lerr = errors
		steps = 201
		chimin = chifunc(M0min, gOVmin, gPNNmin, aOpmin, eOmin, Amin, Bmin, Lmin)
		if tag == 'M0':
			x0 = M0min
			delx = M0err
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(i,gOVmin,gPNNmin,aOpmin,eOmin,Amin,Bmin,Lmin) for i in x])
		elif tag == 'gOV':
			x0 = gOVmin
			delx = gOVerr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,i,gPNNmin,aOpmin,eOmin,Amin,Bmin,Lmin) for i in x])
		elif tag == 'gPNN':
			x0 = gPNNmin
			delx = gPNNerr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,i,aOpmin,eOmin,Amin,Bmin,Lmin) for i in x])
		elif tag == 'aOp':
			x0 = aOpmin
			delx = aOperr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,gPNNmin,i,eOmin,Amin,Bmin,Lmin) for i in x])
		elif tag == 'eO':
			x0 = eOmin
			delx = eOerr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,gPNNmin,aOpmin,i,Amin,Bmin,Lmin) for i in x])
		elif tag == 'A':
			x0 = Amin
			delx = Aerr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,gPNNmin,aOpmin,eOmin,i,Bmin,Lmin) for i in x])
		elif tag == 'B':
			x0 = Bmin
			delx = Berr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,gPNNmin,aOpmin,eOmin,Amin,i,Lmin) for i in x])
		elif tag == 'L':
			x0 = Lmin
			delx = Lerr
			x = np.linspace(x0-delx, x0+delx, steps)
			y = np.array([chifunc(M0min,gOVmin,gPNNmin,aOpmin,eOmin,Amin,Bmin,i) for i in x])
		else:
			print('The tag name is not correct. Please choose from (M0,gOV,gPNN,aOp,eO,A,B,L)')
			return False

		#print(chimin,y[int((steps-1)/2)])
		#print(x[1]-x[0],2*delx/(steps-1))

		index_midpoint = int((steps-1)/2)
		dx = 2*delx/(steps-1)
		dd = (y[index_midpoint+1] + y[index_midpoint-1] - 2*y[index_midpoint])/dx**2
		sigma = np.sqrt(2/dd)
		return x0, sigma, x, y
