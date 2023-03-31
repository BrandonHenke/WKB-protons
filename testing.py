
def test3(**kwargs):
	for key in kwargs.keys():
		print(f"{key}: {kwargs[key]}")

def test2(*args):
	res = 0
	for x in args:
		res += x
	return res

def test(a,*args):
	res = a
	res += test2(*args)
	return res


def main():
	a = 1
	b = 2
	c = 3
	d = test(1,2,3)
	print(d)

if __name__=="__main__":
	main()