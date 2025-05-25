package multidimensional

func bracketMinimum(phi func(float64) float64) (a, b float64) {
	a = 0
	b = 1
	fb := phi(b)
	for fb < phi(b/2) {
		b *= 2
		fb = phi(b)
		if b > 1e6 {
			break
		}
	}
	return a, b
}
