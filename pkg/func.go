package pkg

import "math"

func F1(x float64) float64 {
	return x + 1/(x*x)
}

func DF1(x float64) float64 {
	return 1 - 2/(x*x*x)
}

func DDF1(x float64) float64 {
	return 6 / (x * x * x * x)
}

func F2(x, y float64) float64 {
	return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
}

func GradF2(x, y float64) (gx, gy float64) {
	expv := math.Exp(x*x + y*y)
	gx = 2*x + 2*x*expv + 4
	gy = 2*y*expv + 3
	return
}

func HessF2(x, y float64) (hxx, hxy, hyx, hyy float64) {
	E := math.Exp(x*x + y*y)
	hxx = 2 + 2*E + 4*x*x*E
	hyy = 2*E + 4*y*y*E
	hxy = 4 * x * y * E
	hyx = hxy
	return
}

// g(x_i) >= 0, i = 1,2
// g1(x,y) = x >= 0
// g2(x,y) = y >= 0
func F3(x, y float64) float64 {
	return 9*x*x + y*y - 54*x + 4*y
}

func GradF3(x, y float64) (gx, gy float64) {
	gx = 18*x - 54
	gy = 2*y + 4
	return
}
