package multidimensional

import (
	"math"

	zeroordered "github.com/vshulcz/edu_optimization_methods/internal/1_zero_ordered"
)

func CoordinateDescent2D(
	f func(x, y float64) float64,
	x0, y0, ax, bx, ay, by, eps float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	prevX, prevY := x, y
	prevF := phiF(x, y)

	for {
		gx := func(xx float64) float64 { return phiF(xx, y) }
		x, _, _ = zeroordered.PassiveSearch(gx, ax, bx, eps)

		gy := func(yy float64) float64 { return phiF(x, yy) }
		y, _, _ = zeroordered.PassiveSearch(gy, ay, by, eps)

		currF := phiF(x, y)

		if math.Hypot(x-prevX, y-prevY) < eps || math.Abs(currF-prevF) < eps {
			break
		}

		prevX, prevY, prevF = x, y, currF
	}

	return x, y, phiF(x, y), iters
}

func GradientDescentBacktracking(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, alphaHat, epsilon, lambda, deltaGrad float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		gradNorm2 := gx*gx + gy*gy
		if math.Sqrt(gradNorm2) < deltaGrad {
			break
		}

		alpha := alphaHat
		fx := phiF(x, y)
		var xNew, yNew, fxNew float64
		for {
			xNew = x - alpha*gx
			yNew = y - alpha*gy
			fxNew = phiF(xNew, yNew)
			if fxNew-fx <= -alpha*epsilon*gradNorm2 {
				break
			}
			alpha *= lambda
		}
		x, y = xNew, yNew
	}

	return x, y, phiF(x, y), iters
}

func SteepestDescent(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		phi := func(alpha float64) float64 {
			return phiF(x-alpha*gx, y-alpha*gy)
		}
		a, b := bracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x -= alpha * gx
		y -= alpha * gy
	}

	return x, y, phiF(x, y), iters
}

func AcceleratedGradientDescent(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0 float64,
	p int,
	gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		xs, ys := x, y
		for range p {
			gxs, gys := grad(xs, ys)
			phi1 := func(alpha float64) float64 {
				return phiF(xs-alpha*gxs, ys-alpha*gys)
			}
			a, b := bracketMinimum(phi1)
			alpha, _, _ := zeroordered.GoldenSectionSearch(phi1, a, b, gradEps)
			xs -= alpha * gxs
			ys -= alpha * gys
		}

		dx, dy := xs-x, ys-y
		phi2 := func(alpha float64) float64 {
			return phiF(x+alpha*dx, y+alpha*dy)
		}
		a, b := bracketMinimum(phi2)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi2, a, b, gradEps)

		x += alpha * dx
		y += alpha * dy
	}

	return x, y, phiF(x, y), iters
}

func RavineStep(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, delta float64,
	p int,
	gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		xt, yt := x+delta, y+delta

		xs, ys := x, y
		for range p {
			gxs, gys := grad(xs, ys)
			phi := func(alpha float64) float64 {
				return phiF(xs-alpha*gxs, ys-alpha*gys)
			}
			a, b := bracketMinimum(phi)
			alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)
			xs -= alpha * gxs
			ys -= alpha * gys
		}

		xst, yst := xt, yt
		for range p {
			gxst, gyst := grad(xst, yst)
			phi := func(alpha float64) float64 {
				return phiF(xst-alpha*gxst, yst-alpha*gyst)
			}
			a, b := bracketMinimum(phi)
			alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)
			xst -= alpha * gxst
			yst -= alpha * gyst
		}

		dx, dy := xst-xs, yst-ys

		phi := func(alpha float64) float64 {
			return phiF(xs+alpha*dx, ys+alpha*dy)
		}
		a, b := bracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x = xs + alpha*dx
		y = ys + alpha*dy
	}

	return x, y, phiF(x, y), iters
}

func NewtonModified(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	hess func(x, y float64) (hxx, hxy, hyx, hyy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		hxx, hxy, hyx, hyy := hess(x, y)
		det := hxx*hyy - hxy*hyx
		if math.Abs(det) < 1e-14 {
			panic("Hessian is singular")
		}

		px := -(hyy*gx - hxy*gy) / det
		py := -(-hyx*gx + hxx*gy) / det

		phi := func(alpha float64) float64 {
			return phiF(x+alpha*px, y+alpha*py)
		}
		a, b := bracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x += alpha * px
		y += alpha * py
	}

	return x, y, phiF(x, y), iters
}

func QuasiNewton(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	Hxx, Hxy := 1.0, 0.0
	Hyx, Hyy := 0.0, 1.0

	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		px := -(Hxx*gx + Hxy*gy)
		py := -(Hyx*gx + Hyy*gy)

		phi := func(alpha float64) float64 {
			return phiF(x+alpha*px, y+alpha*py)
		}
		a, b := bracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		xNew := x + alpha*px
		yNew := y + alpha*py

		dx := xNew - x
		dy := yNew - y
		gx2, gy2 := grad(xNew, yNew)
		gammaX := gx2 - gx
		gammaY := gy2 - gy

		v_x := dx - (Hxx*gammaX + Hxy*gammaY)
		v_y := dy - (Hyx*gammaX + Hyy*gammaY)

		denom := v_x*gammaX + v_y*gammaY
		if math.Abs(denom) > 1e-14 {
			Hxx += v_x * v_x / denom
			Hxy += v_x * v_y / denom
			Hyx += v_y * v_x / denom
			Hyy += v_y * v_y / denom
		}
		if iters%2 == 0 {
			Hxx, Hxy = 1.0, 0.0
			Hyx, Hyy = 0.0, 1.0
		}

		x, y = xNew, yNew
	}

	return x, y, phiF(x, y), iters
}

func ConjGradFR(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	// текущее приближение
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	gx, gy := grad(x, y)
	dx, dy := -gx, -gy

	for {
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		phi := func(alpha float64) float64 {
			return phiF(x+alpha*dx, y+alpha*dy)
		}
		a, b := bracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		// x_{k+1} = x_k + α_k d_k
		x += alpha * dx
		y += alpha * dy

		gxNew, gyNew := grad(x, y)

		// Метод Флетчера-Ривза: β_k = (‖g_{k+1}‖²)/(‖g_k‖²)
		num := gxNew*gxNew + gyNew*gyNew
		den := gx*gx + gy*gyNew

		var beta float64
		if den > 0 {
			beta = num / den
		}

		// d_{k+1} = -g_{k+1} + β_k d_k
		dx = -gxNew + beta*dx
		dy = -gyNew + beta*dy

		gx, gy = gxNew, gyNew
	}

	xmin, ymin = x, y
	fmin = phiF(x, y)

	return xmin, ymin, fmin, iters
}
