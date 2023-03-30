import numpy as np



def roots(func,r0,args):
	if not hasattr(r0, '__iter__'):
		r0 = [r0]
	
	dr = 0.1
	points = np.array(r0)

	for n in range(len(points)):
		sign = func(points[n],*args) > 0
		r = points[n]
		while (func(r,*args) > 0) == sign:
			r += dr
		
		while np.abs(func(r,*args)) >= 10**(-10):
			df = (func(r+dr,*args)-func(r,*args))/dr
			r = r - (func(r,*args)/df)
		
		points[n] = r

	return list(points)

def V(r):
	return r**2

def V_E0(r,E_0):
	return V(r)-E_0

def main():
	E_0 = 1
	r = roots(V_E0,10**(-2),args=(E_0,))

	print(f"{r}")

if __name__ == "__main__":
	main()
 