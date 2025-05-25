package zeroordered

import (
	"math"
)

func PassiveSearch(f func(x float64) float64, a, b, eps float64) (xmin, fmin float64, iterations int) {
	k := int(math.Ceil((b - a) / eps))

	minVal := math.Inf(1)
	minX := a

	for i := 0; i <= k; i++ {
		x := a + float64(i)*(b-a)/float64(k)
		val := f(x)
		if val < minVal {
			minVal = val
			minX = x
		}
		iterations++
	}

	return minX, minVal, iterations
}

func DichotomySearch(f func(x float64) float64, a, b, eps, delta float64) (xmin, fmin, aFinal, bFinal float64, iterations int) {
	for (b-a)/2.0 > eps {
		mid := (a + b) / 2.0
		c := mid - delta/2.0
		d := mid + delta/2.0

		iterations += 2
		if f(c) <= f(d) {
			b = d
		} else {
			a = c
		}
	}
	xmin = (a + b) / 2.0
	fmin = f(xmin)
	aFinal = a
	bFinal = b
	return
}

func GoldenSectionSearch(f func(x float64) float64, a, b, eps float64) (xmin, fmin float64, i int) {
	ad := (math.Sqrt(5) - 1) / 2
	ac := (3 - math.Sqrt(5)) / 2

	c := a + ac*(b-a)
	d := a + ad*(b-a)
	fc := f(c)
	fd := f(d)

	i = 2
	for (b-a)/2.0 > eps {
		if fc <= fd {
			b = d
			d = c
			fd = fc
			c = a + ac*(b-a)
			fc = f(c)
		} else {
			a = c
			c = d
			fc = fd
			d = a + ad*(b-a)
			fd = f(d)
		}
		i++
	}

	xmin = (a + b) / 2.0
	fmin = f(xmin)
	return
}

func FibonacciSearch(f func(x float64) float64, a, b, eps float64) (xmin, fmin float64, iterations int) {
	fib := []float64{1, 1}
	var n int
	for i := 2; ; i++ {
		fib = append(fib, fib[i-1]+fib[i-2])
		if fib[i] >= (b-a)/eps {
			n = i
			break
		}
	}
	if n < 3 {
		n = 3
	}

	iterations = 0
	eval := func(x float64) float64 {
		iterations++
		return f(x)
	}

	a_n, b_n := a, b
	x1 := a_n + fib[n-2]/fib[n]*(b_n-a_n)
	x2 := a_n + fib[n-1]/fib[n]*(b_n-a_n)
	f1 := eval(x1)
	f2 := eval(x2)

	iterations = 1
	for k := 1; k <= n-3; k++ {
		if f1 > f2 {
			a_n = x1
			x1 = x2
			f1 = f2
			x2 = a_n + fib[n-k-1]/fib[n-k]*(b_n-a_n)
			f2 = eval(x2)
		} else {
			b_n = x2
			x2 = x1
			f2 = f1
			x1 = a_n + fib[n-k-2]/fib[n-k]*(b_n-a_n)
			f1 = eval(x1)
		}
	}

	delta := eps / 10.0
	x2 = x1 + delta
	f2 = eval(x2)
	if f1 > f2 {
		a_n = x1
	} else {
		b_n = x2
	}

	xmin = (a_n + b_n) / 2.0
	fmin = f(xmin)
	return
}
