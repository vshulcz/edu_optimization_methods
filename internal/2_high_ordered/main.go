package highordered

import (
	"math"
)

func TangentSearch(
	f func(x float64) float64,
	df func(x float64) float64,
	a, b, eps float64,
) (xmin, fmin float64, iters int) {
	phiF := func(x_ float64) float64 {
		iters++
		return f(x_)
	}

	fa, fb := df(a), df(b)
	if fa >= 0 {
		return a, phiF(a), 0
	}
	if fb <= 0 {
		return b, phiF(b), 0
	}

	var x0 float64
	for {
		m1 := df(a)
		m2 := df(b)
		c1 := phiF(a) - m1*a
		c2 := phiF(b) - m2*b

		x0 = (c2 - c1) / (m1 - m2)
		dfx0 := df(x0)

		if math.Abs(b-a) <= eps || math.Abs(dfx0) <= eps {
			break
		}

		if dfx0 > 0 {
			b = x0
		} else {
			a = x0
		}
	}
	xmin = x0
	fmin = phiF(x0)
	return
}

func NewtonSearch(
	f func(x float64) float64,
	df func(x float64) float64,
	d2f func(x float64) float64,
	x0, eps float64,
) (xmin, fmin float64, iters int) {
	phiF := func(x_ float64) float64 {
		iters++
		return f(x_)
	}

	x := x0
	for {
		g := df(x)
		if math.Abs(g) <= eps {
			break
		}
		h := d2f(x)
		if h == 0 {
			break
		}
		x = x - g/h
	}
	xmin = x
	fmin = phiF(x)
	return
}

func SecantSearch(
	f func(x float64) float64,
	df func(x float64) float64,
	x0, x1, eps float64,
) (xmin, fmin float64, iters int) {
	phiF := func(x_ float64) float64 {
		iters++
		return f(x_)
	}

	f0 := df(x0)
	for {
		f1 := df(x1)
		if math.Abs(f1) <= eps {
			xmin = x1
			break
		}
		denom := f1 - f0
		if denom == 0 {
			break
		}
		x2 := x1 - (x1-x0)*f1/denom
		x0, f0 = x1, f1
		x1 = x2
	}
	xmin = x1
	fmin = phiF(x1)
	return
}
