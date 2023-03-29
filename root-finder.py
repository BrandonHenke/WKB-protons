import numpy as np

def V(r):
	return np.sin(r)

def turns(E_0,r0):
	dr = 0.1
	points = [r0]*3
	print(points)

	for n in range(3):
		if n >= 1:
			points[n] = points[n-1]+dr
		
		sign = (V(points[n]) - E_0 > 0)
		r = points[n]
		while (V(r) - E_0 > 0) == sign:
			r += dr
		
		while np.abs(V(r) - E_0) >= 10**(-4):
			df = (V(r+dr)-V(r))/dr
			r = r - (V(r) - E_0)/df

		points[n] = r

	if len(points) == 3:
		return points[0],points[1],points[2]

def main():
	r0,r1,r2 = turns(0,10**(-2))

	print(f"{r0}, {r1}, {r2}")

if __name__ == "__main__":
	main()
