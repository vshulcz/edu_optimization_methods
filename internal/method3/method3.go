package method3

import "math"

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
