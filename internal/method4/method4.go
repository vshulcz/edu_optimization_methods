package method4

func FibonacciSearch(f func(x float64) float64, a, b, eps float64) (xmin, fmin float64, iterations int) {
	fib := []float64{0, 1, 1}
	for fib[len(fib)-1] < (b-a)/eps {
		nextFib := fib[len(fib)-1] + fib[len(fib)-2]
		fib = append(fib, nextFib)
	}
	m := len(fib) - 1
	n := m - 2

	c := a + (b-a)*(fib[n]/fib[n+2])
	d := a + (b-a)*(fib[n+1]/fib[n+2])
	fc := f(c)
	fd := f(d)

	iterations = 1
	for i := 1; i <= n-1; i++ {
		if fc > fd {
			a = c
			c = d
			fc = fd
			d = a + (b-a)*(fib[n+1-i]/fib[n+2-i])
			fd = f(d)
		} else {
			b = d
			d = c
			fd = fc
			c = a + (b-a)*(fib[n-i]/fib[n+2-i])
			fc = f(c)
		}
		iterations++
	}

	xmin = (a + b) / 2.0
	fmin = f(xmin)
	return
}
