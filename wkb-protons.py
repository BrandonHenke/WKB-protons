import numpy as np
import matplotlib.pyplot as plt


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
			6,
			6,
			6,
			3,
			6
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

V_0		= 54 # MeV
V_SO	= 0.2 * V_0
e2		= 1.4399764 # MeV fm
a		= 0.7 # fm
ħ		= 197.3269804 # MeV fm
m_π		= 139.570 # MeV
ħ_mπ	= 2.044 # fm^2
μ		= 938.2720813 # MeV
c		= 2.99792458e+23

def V_Coul(r,A,Z):
	R = 1.2 * A**(1/3) # fm
	# print(f"R = {R}")

	R0 = r[r>R]
	R1 = r[r<=R]

	return np.concatenate((
			Z*e2/(2*R) * (3-(R1/R)**2),
			Z*e2/R0
		))

	# if r > R:
	# 	return Z*e2/r
	# else:
	# 	return Z*e2/(2*R) * (3-(r/R)**2)

def V_WS(r,A,j,l):
	R		= 1.2 * A**(1/3) # fm

	fws		= (1+np.exp((r-R)/a))**(-1)
	dfws	= 1/( -2*a*np.cosh((R - r)/a) - 2*a)

	s = j-l
	if s > 0:	l_dot_s = l*s
	else:		l_dot_s = -(l+1)*s

	return -V_0*fws + V_SO * ħ_mπ * (2/r) * dfws * l_dot_s

def V(r,A,Z,j,l):
	r = np.array(r)
	return V_WS(r,A,j,l) + V_Coul(r,A,Z) + ħ**2/(2*μ*r**2) * l * (l+1)

def k(r,A,Z,j,l,E_0):
	return np.emath.sqrt(2*μ*(E_0 - V(r,A,Z,j,l)))/ħ

def turns(r0,A,Z,j,l,E_0):
	dr = 0.1
	points = [r0]*3

	for n in range(3):
		if n >= 1:
			points[n] = points[n-1]+dr
		
		sign = V(points[n],A,Z,j,l) - E_0 > 0
		r = points[n]
		while (V(r,A,Z,j,l) - E_0 > 0) == sign:
			r += dr
		
		# print(n)
		# print(f"V0 = {V(r-dr,A,Z,j,l)}, E0 = {E_0}")
		# print(f"V0 - E0 = {V(r-dr,A,Z,j,l)-E_0}")
		# print(f"V = {V(r,A,Z,j,l)}, E0 = {E_0}")
		# print(f"V - E0 = {V(r,A,Z,j,l)-E_0}")
		# print()

		while np.abs(V(r,A,Z,j,l) - E_0) >= 10**(-3):
			df = (V(r+dr,A,Z,j,l)-V(r,A,Z,j,l))/dr
			r = r - ((V(r,A,Z,j,l) - E_0)/df)[0]
		
		points[n] = r

	return points[0],points[1],points[2]

def Γ(Sp,A,Z,j,l,E_0):

	r0,r1,r2 = turns(0.001,A,Z,j,l,E_0)
	# print(f"{r0}, {r1}, {r2}")
	dr = 0.01
	regA = np.arange(r0,r1,dr)
	regB = np.arange(r1,r2,dr)
	r = np.concatenate((
		np.arange(10**(-3),r0,dr),
		regA,
		regB,
		np.arange(r2,1.5*r2,dr),
	))
	kA = k(regA,A,Z,j,l,E_0)
	kB = k(regB,A,Z,j,l,E_0)
	integral1 = np.trapz(1/kA[np.abs(np.imag(kA)) <= 10**(-5)],regA[np.abs(np.imag(kA)) <= 10**(-5)]).real
	# integral1 = np.trapz(1/kA,regA)
	integral2 = np.trapz(np.abs(kB),regB)

	N = (1/2 * integral1)**(-1)
	return Sp * N * ħ**2/(4*μ) * np.exp(-2 * integral2), r

def main():
	Sp = 1
	for n in range(A.shape[1]):
		γ,r = Γ(Sp,A[0,n],Z[0,n],j[0,n],l[0,n],E_0[0,n])
		τ_12 = ħ*np.log(2)/(γ*c)
		print(f"n = {n}, T_1/2 = {τ_12*10**6} μs")
	# r = np.arange(0,40,0.01)
	y = V(r,A[0,0],Z[0,0],j[0,0],l[0,0])
	fig = plt.figure()
	plt.plot(r,y,label=r"$V(r)$")
	plt.plot(r,[E_0[0,0]]*len(r),label=r"$E_0$")
	plt.ylim(np.nanmin(y[y != -np.inf]),-np.nanmin(y[y != -np.inf]))
	plt.title("Iodine")
	plt.xlabel("r (fm)")
	plt.ylabel("Energy (MeV)")
	plt.legend()
	plt.savefig("test.png")

if __name__ == "__main__":
	main()