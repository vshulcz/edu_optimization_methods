package method2

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
