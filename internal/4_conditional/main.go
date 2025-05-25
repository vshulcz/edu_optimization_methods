package conditional

import "math"

func KuhnTucker(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	gradEps float64,
) (xmin, ymin, fmin, lam1Opt, lam2Opt float64, iters int) {
	// Градиент в точке (0,0)
	g0x, g0y := grad(0, 0)
	// Градиент в точке (h,0) и (0,h)
	g1x, g1y := grad(gradEps, 0)
	g2x, g2y := grad(0, gradEps)

	// Приближённые компоненты гессиана:
	// H_{ij} ≈ (∂f/∂x_i (x+he_j) − ∂f/∂x_i (x)) / h
	H00 := (g1x - g0x) / gradEps // ∂²f/∂x²
	H01 := (g2x - g0x) / gradEps // ∂²f/∂y∂x
	H10 := (g1y - g0y) / gradEps // ∂²f/∂x∂y
	H11 := (g2y - g0y) / gradEps // ∂²f/∂y²
	c0, c1 := g0x, g0y

	// Перебор всех возможных активных множеств
	// 2^2=4 комбинации активных (g_i(x)=0) / неактивных (λ_i=0).
	for mask := range 4 {
		iters++

		// Переменные [x, y, λ1, λ2] → 4 уравнения
		const N = 4
		A := make([]float64, N*N)
		b := make([]float64, N)

		// Стационарность ∇f(x,y) − λ1∇g1 − λ2∇g2 = 0
		// g1(x,y)=x => ∇g1=(1,0), g2(x,y)=y => ∇g2=(0,1)
		// H00*x + H01*y   - 1*λ1        = -c0
		// H10*x + H11*y         - 1*λ2   = -c1

		// уравнение по x
		A[0*N+0], A[0*N+1], A[0*N+2] = H00, H01, -1
		b[0] = -c0

		// уравнение по y
		A[1*N+0], A[1*N+1], A[1*N+3] = H10, H11, -1
		b[1] = -c1

		// Для i=1 (x>=0):
		//    если i-е ограничение активно (mask&1≠0) → добавить уравнение g1(x,y)=x=0
		//    иначе → добавить λ1=0
		if mask&1 != 0 {
			// x = 0
			A[2*N+0] = 1
			b[2] = 0
		} else {
			// λ1 = 0
			A[2*N+2] = 1
			b[2] = 0
		}

		// Для i=2 (y>=0):
		//    если mask&2≠0 → y = 0, иначе λ₂ = 0
		if mask&2 != 0 {
			A[3*N+1] = 1
			b[3] = 0
		} else {
			A[3*N+3] = 1
			b[3] = 0
		}

		// Решаем СЛАУ условий куна таккера методом Гаусса
		sol, err := SolveGauss(A, b, N)
		if err != nil {
			continue
		}
		x, y, l1, l2 := sol[0], sol[1], sol[2], sol[3]

		if x < -gradEps || y < -gradEps || l1 < -gradEps || l2 < -gradEps {
			continue
		}

		xmin, ymin, lam1Opt, lam2Opt = x, y, l1, l2
		fmin = f(x, y)
		return
	}
	return
}

func ExternalPenalty(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0 float64,
	r0, rFactor, epsConstr, epsGrad, alpha float64,
	maxOuterIter, maxInnerIter int,
) (xmin, ymin, fmin, r float64, outerIt, innerIt int) {
	// Штрафная функция H(x,y) = [max(0, -x)]^2 + [max(0, -y)]^2,
	// где g1(x)= -x <= 0  эквивалентно x >= 0, g2(y)= -y <= 0
	H := func(x, y float64) float64 {
		h1 := math.Max(0, -x)
		h2 := math.Max(0, -y)
		return h1*h1 + h2*h2
	}
	// И её градиент:
	// ∂H/∂x = { 2x, если x<0; 0, иначе }
	// ∂H/∂y = { 2y, если y<0; 0, иначе }
	gradH := func(x, y float64) (hx, hy float64) {
		if x < 0 {
			hx = 2 * x
		}
		if y < 0 {
			hy = 2 * y
		}
		return
	}

	xmin, ymin = x0, y0
	r = r0

	// Внешний цикл по штрафам
	for k := 1; k <= maxOuterIter; k++ {
		outerIt = k

		// φ_k(x) = f(x) + r * H(x)
		// Простой градиентный спуск с фиксированным шагом alpha
		for it := 1; it <= maxInnerIter; it++ {
			innerIt = it

			// градиент f
			dfdx, dfdy := grad(xmin, ymin)
			// градиент штрафа
			hdx, hdy := gradH(xmin, ymin)

			// градиент φ
			gpx := dfdx + r*hdx
			gpy := dfdy + r*hdy

			// Признак остановка внутреннего метода
			if math.Hypot(gpx, gpy) <= epsGrad {
				break
			}

			// шаг вперед
			xmin -= alpha * gpx
			ymin -= alpha * gpy
		}

		// Допустимость H(x) ≤ ε
		if H(xmin, ymin) <= epsConstr {
			break
		}
		// иначе — увеличиваем штраф и повторяем
		r *= rFactor
	}

	fmin = f(xmin, ymin)

	return
}
