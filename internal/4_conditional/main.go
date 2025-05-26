package conditional

import (
	zeroordered "github.com/vshulcz/edu_optimization_methods/internal/1_zero_ordered"
	multidimensional "github.com/vshulcz/edu_optimization_methods/internal/3_multidimensional"
	"github.com/vshulcz/edu_optimization_methods/pkg"
)

// KuhnTucker решает:
//
//	min f(x,y)  при  x ≥ 0, y ≥ 0
//
// с помощью условий Куна–Таккера.
//
// Алгоритм:
// 1) Угол (x=0,y=0):
//   - Вычисляем g0 = ∇f(0,0).
//   - Если g0.x ≥ 0 и g0.y ≥ 0, то минимум в (0,0), λ1=g0.x, λ1=g0.y.
//
// 2) Граница x=0 (только λ1 активен):
//   - Если ∂f/∂y(0,0) = g0.y < 0, функция убывает вдоль y>0.
//   - Решаем одномерную задачу  min_{y≥0} f(0,y) (bracket + GoldenSection).
//   - Получаем y*, проверяем λ1 = ∂f/∂x(0,y*) ≥ 0 и λ2=0.
//
// 3) Граница y=0 (только λ2 активен) – аналогично:
//   - Если g0.x < 0, минимизируем f(x,0) по x≥0,
//   - Затем λ2 = ∂f/∂y(x*,0) ≥ 0, λ1=0.
//
// 4) Никакое ограничение не активно:
//   - Если g0.x < 0 и g0.y < 0, решаем ∇f(x,y)=0 квазиньютоновским методом (QuasiNewton).
//   - Проверяем x≥0, y≥0, λ1=λ2=0.
//
// Параметры:
// - f: целевая функция двух переменных.
// - grad: её градиент (gx, gy).
// - eps: точность для одномерных и многомерных методов.
//
// Возвращает:
// - xmin, ymin: координаты найденного минимума.
// - fmin: значение f в этой точке.
// - lam1Opt, lam2Opt: оптимальные множители Лагранжа.
// - iters: число вызовов f (для оценки вычислительных затрат).func KuhnTucker(
func KuhnTucker(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	eps float64,
) (xmin, ymin, fmin, lam1Opt, lam2Opt float64, iters int) {
	phiF := func(x, y float64) float64 {
		iters++
		return f(x, y)
	}

	// проверка угла (0,0)
	gx0, gy0 := grad(0, 0)
	if gx0 >= 0 && gy0 >= 0 {
		return 0, 0, phiF(0, 0), gx0, gy0, iters
	}

	// граница x=0, φ(y)=f(0,y), φ'(0)=gy0
	if gy0 < 0 {
		phiY := func(y float64) float64 { return phiF(0, y) }
		a, b := pkg.BracketMinimum(phiY)
		y1, fy1, _ := zeroordered.GoldenSectionSearch(phiY, a, b, eps)
		if y1 >= 0 {
			gx1, _ := grad(0, y1)
			if gx1 >= 0 {
				return 0, y1, fy1, gx1, 0, iters
			}
		}
	}

	// граница y=0, φ(x)=f(x,0), φ'(0)=gx0
	if gx0 < 0 {
		phiX := func(x float64) float64 { return phiF(x, 0) }
		a, b := pkg.BracketMinimum(phiX)
		x1, fx1, _ := zeroordered.GoldenSectionSearch(phiX, a, b, eps)
		if x1 >= 0 {
			_, gy1 := grad(x1, 0)
			if gy1 >= 0 {
				return x1, 0, fx1, 0, gy1, iters
			}
		}
	}

	// gx0<0 и gy0<0
	x0, y0, f0, _ := multidimensional.QuasiNewton(phiF, grad, 0, 0, eps)
	if x0 >= 0 && y0 >= 0 {
		return x0, y0, f0, 0, 0, iters
	}

	return
}

// ExternalPenalty решает задачу
// min f(x,y)  при  x ≥ 0, y ≥ 0
// с помощью метода внешних штрафов.
//
// Идея: каждое ограничение g_i(x) ≤ 0 «штрафуется» непрерывной
// функцией H(x) = Σ[max(0, g_i(x))]², здесь g₁=−x, g₂=−y.
// При большом коэффициенте штрафа r минимум
// φ_r(x)=f(x)+r·H(x) лежит почти в допустимой области.
//
// Алгоритм:
//  1. Задаём (x,y)=(x0,y0), r=r0.
//  2. Повторяем (до maxOuterIter):
//     a) Безусловно минимизируем φ_r(x,y) методом
//     наискорейшего градиентного спуска со стоп-критерием
//     ‖∇φ_r‖≤epsGrad.
//     b) Если H(x,y)≤epsConstr — принимаем решение.
//     c) Иначе r *= rFactor и возвращаемся к (a).
//
// Параметры:
// - f, grad    : цель и её градиент.
// - x0,y0      : стартовая точка.
// - r0         : начальный штраф.
// - rFactor    : во сколько раз увеличиваем r при невыполнении H≤epsConstr.
// - epsConstr  : допустимый уровень штрафа для завершения.
// - epsGrad    : точность внутреннего градиентного спуска.
// - maxOuterIt : макс. число штрафных итераций.
//
// Возвращает:
// - xmin,ymin  : найденный минимум.
// - fmin       : f(xmin,ymin).
// - r          : финальный коэффициент штрафа.
// - outerIt    : сколько раз обновляли r.
func ExternalPenalty(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0 float64,
	r0, rFactor, epsConstr, epsGrad float64,
	maxOuterIter int,
) (xmin, ymin, fmin, r float64, outerIt int) {
	// Штрафная функция H(x,y) = [max(0, -x)]^2 + [max(0, -y)]^2,
	// где g1(x)= -x <= 0  эквивалентно x >= 0, g2(y)= -y <= 0
	H := func(x, y float64) float64 {
		h1 := max(0, -x)
		h2 := max(0, -y)
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
	// φ_k(x) = f(x) + r * H(x)
	// Градиентный спуск
	for outerIt = 1; outerIt <= maxOuterIter; outerIt++ {
		// φ_r(x,y) = f + r·H
		phiF := func(x, y float64) float64 {
			return f(x, y) + r*H(x, y)
		}
		gradPhiF := func(x, y float64) (gx, gy float64) {
			dfx, dfy := grad(x, y)
			hx, hy := gradH(x, y)
			return dfx + r*hx, dfy + r*hy
		}

		xNew, yNew, _, _ := multidimensional.SteepestGradientDescent(
			phiF, gradPhiF,
			xmin, ymin,
			epsGrad,
		)
		xmin, ymin = xNew, yNew

		// Допустимость H(x) ≤ ε
		if H(xmin, ymin) <= epsConstr {
			break
		}
		// иначе — увеличиваем штраф и повторяем
		r *= rFactor
	}

	fmin = f(xmin, ymin)
	return xmin, ymin, fmin, r, outerIt
}
