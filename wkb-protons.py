import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


E_0		= np.array([[
			829,
			823,
			977,
			1140,
			1210,
			1071,
			1132,
			1283
		]]) * 10**(-3) # MeV
A		= np.array([[
			109,
			112,
			113,
			146,
			146,
			147,
			147,
			150
		]])
Z		= np.array([[
			53,
			55,
			55,
			69,
			69,
			69,
			69,
			71
		]])
l		= np.array([[
			3,
			3,
			3,
			5,
			5,
			5,
			3,
			5
		]])
j		= np.array([[
			5/2,
			5/2,
			5/2,
			11/2,
			11/2,
			11/2,
			3/2,
			11/2
		]])
t_12	= np.array([[
			100e-6,
			500e-6,
			17e-6,
			235e-3,
			72e-3,
			2.7,
			360e-6,
			40e-3
		]])

V_0		= 54 # MeV
V_SO	= 0.2 * V_0
e2		= 1.4399764 # MeV fm
a		= 0.7 # fm
ħ		= 197.3269804 # MeV fm
m_π		= 139.570 # MeV
ħ_mπ	= 2.044 # fm^2
μ		= 938.2720813 # MeV
m_n		= 939.5654205 # MeV
c		= 2.99792458e+23 # fm/s

def V_Coul(r,A,Z):
	R = 1.2 * A**(1/3) # fm
	# print(f"R = {R}")

	R0 = r[r>R]
	R1 = r[r<=R]

	return np.concatenate((
			Z*e2/(2*R) * (3-(R1/R)**2),
			Z*e2/R0
		)).flatten()

def V_WS(r,A,j,l):
	# print(A)
	R		= 1.2 * A**(1/3) # fm
	
	fws		= (1+np.exp((r-R)/a))**(-1)
	dfws	= (-2*a*(np.cosh((R - r)/a) + 1))**(-1)

	s = j-l
	if s > 0:	l_dot_s = l*s
	else:		l_dot_s = -(l+1)*s

	return -V_0*fws + V_SO * ħ_mπ * (2/r) * dfws * l_dot_s

def V(r,A,Z,j,l):
	r = np.array(r)
	return V_WS(r,A,j,l) + V_Coul(r,A,Z) + ħ**2/(2*μ*r**2) * l * (l+1)

def V_neutron(r,A,j,l):
	r = np.array(r)
	return V_WS(r,A,j,l) + ħ**2/(2*μ*r**2) * l * (l+1)

def k(r,W,E_0,**kwargs):
	args = kwargs["args"]
	return np.emath.sqrt(2*μ*(E_0 - W(r,*args)))/ħ

def V_E0(r,W,E_0,*args):
	return W(r,*args)-E_0

def roots(func,r0,n,**kwargs):
	args = kwargs["args"]
	dr = 0.1
	points = [r0]*n
	
	for n in range(len(points)):
		if n >= 1:
			points[n] = points[n-1] + dr
		
		sign = func(points[n],*args) > 0
		r = points[n]
		while (func(r,*args) > 0) == sign:
			r += dr
		
		while np.abs(func(r,*args)) >= 10**(-10):
			df = (func(r+dr,*args)-func(r,*args))/dr
			r = r - (func(r,*args)/df)
		
		points[n] = r

	return points


def Γ(Sp,W,E_0,*args):
	# print(E_0)
	dr = 0.01
	rr = roots(V_E0,0.1,3,args=(W,E_0,*args))

	regA = np.arange(rr[0],rr[1]+dr,dr)
	regB = np.arange(rr[1],rr[2]+dr,dr)
	r = np.concatenate((
		np.arange(10**(-3),rr[0],dr),
		regA,
		regB,
		np.arange(rr[2],1.5*rr[2],dr),
	))

	kA = k(regA,W,E_0,args=(*args,))
	kB = k(regB,W,E_0,args=(*args,))
	inds = (np.abs(kA.imag) <= 10**(-7)) * (np.abs(kA) >= 10**(-5))
	integral1 = np.trapz(1/(kA[inds]).real,regA[inds])
	integral2 = np.trapz(np.abs(kB),regB)
	
	N = (1/2 * integral1)**(-1)
	γ = Sp * N * ħ**2/(4*μ) * np.exp(-2 * integral2)

	return γ, r

def barrierMax(W,*args):
	dr = 0.1
	r = roots(V_E0,0.1,2,args=(W,0,*args))
	r = r[1]
	while W(r,*args) > W(r-dr,*args):
		r += dr
	maxE = W(r-dr,*args)

	return maxE

def main():
	Sp	= 1
	
	γ1	= np.zeros(A.shape[1])
	r	= [None]*A.shape[1]
	for n in range(A.shape[1]):
		γ1[n],r[n] = Γ(Sp,V,E_0[0,n],A[0,n],Z[0,n],j[0,n],l[0,n])

	τ_12 = ħ*np.log(2)/(γ1*c)
	for n in range(len(τ_12)):
		print(f"A = {A[0,n]}, Z = {Z[0,n]}, T_1/2 = {'%.2E'%τ_12[n]} s")

	maxE = barrierMax(V,A[0,5],Z[0,5],j[0,6],l[0,6])
	dq = 10**(-1)
	Q = np.arange(1,maxE-10*dq,dq)
	γ2 = np.zeros((L:=6,len(Q)))
	for m in range(L):
		for n in range(len(Q)):
			γ2[m,n],r = Γ(Sp,V,Q[n],A[0,5],Z[0,5],j[0,6],l[0,6])
	
	τ_12 = ħ*np.log(2)/(γ2*c)
	

	fig = plt.figure()
	for m in range(L):
		plt.plot(Q,τ_12[m],label=f"l = {m}")
	plt.title(r"$^{147}$Tm")
	plt.xlabel(r"$Q_p$ (MeV)")
	plt.ylabel("Half life (s)")
	plt.yscale("log")
	plt.legend()
	plt.savefig("prob_2.png")
	
	Sp = (ħ*np.log(2)/(t_12*γ1*c))[0]
	for n in range(len(Sp)):
		print(f"A = {A[0,n]}, Z = {Z[0,n]}, Sp = {Sp[n]}")

	Sp = 1
	maxE = barrierMax(V_neutron,150,11/2,5)
	Q = np.arange(1,maxE-10*dq,dq)
	γ3 = np.zeros(len(Q))
	for n in range(len(Q)):
		γ3[n],r = Γ(Sp,V_neutron,Q[n],150,11/2,5)
	τ_12 = ħ*np.log(2)/(γ3*c)
	fig = plt.figure()
	plt.plot(Q,τ_12)
	plt.title(r"$^{150}Z$")
	plt.xlabel(r"$Q_p$ (MeV)")
	plt.ylabel("Half life (s)")
	plt.yscale("log")
	plt.savefig("prob_3.png")
	



if __name__ == "__main__":
	main()