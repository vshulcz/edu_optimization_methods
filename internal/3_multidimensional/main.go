package multidimensional

import (
	"math"

	zeroordered "github.com/vshulcz/edu_optimization_methods/internal/1_zero_ordered"
	"github.com/vshulcz/edu_optimization_methods/pkg"
)

// CoordinateDescent реализует метод покоординатного спуска для минимизации функции двух переменных f(x, y)
// с использованием одномерного метода (например, золотого сечения) на каждом шаге.
//
// Алгоритм:
// - Начинается с точки (x0, y0).
// - По очереди минимизируется функция сначала по x при фиксированном y, затем по y при фиксированном x.
// - Одномерная минимизация на соответствующем отрезке [ax, bx] или [ay, by].
// - Цикл повторяется до достижения условия остановки: изменения координат или значения функции меньше eps.
//
// Особенности:
// - Работает итеративно, проходя по каждой переменной поочерёдно (по осям координат).
// - На каждом шаге решается задача одномерной оптимизации.
// - Прост в реализации, но может "застревать" в неэкстремальных точках.
// - Эффективность зависит от согласованности направления градиента и координатных осей.
//
// Возвращает координаты точки минимума (xmin, ymin), значение функции в этой точке (fmin),
// и общее число вызовов функции f (iters).
func CoordinateDescent(
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
		x, _, _ = zeroordered.GoldenSectionSearch(gx, ax, bx, eps)

		gy := func(yy float64) float64 { return phiF(x, yy) }
		y, _, _ = zeroordered.GoldenSectionSearch(gy, ay, by, eps)

		currF := phiF(x, y)
		if math.Hypot(x-prevX, y-prevY) <= eps || math.Abs(currF-prevF) <= eps {
			break
		}

		prevX, prevY, prevF = x, y, currF
	}

	return x, y, phiF(x, y), iters
}

// GradientDescentBacktracking реализует градиентный метод минимизации функции f(x, y)
// с дроблением шага, основанный на проверке убывания функции по специальному критерию.
//
// На каждой итерации алгоритм ищет шаг α, для которого выполняется условие:
// f(x) ≤ f(x - α ∇f(x)) - α * ε * ||∇f(x)||²
// где ε > 0 — заданный параметр, отвечающий за допустимое уменьшение.
//
// Алгоритм:
//   - Начальная точка: (x0, y0).
//   - На каждой итерации рассчитывается градиент grad(x, y).
//   - Если его норма меньше порога deltaGrad — итерации прекращаются (точка стационарна).
//   - Иначе выбирается начальный шаг alpha = alphaHat и производится "дробление" —
//     шаг уменьшается в alpha *= lambda до тех пор, пока не выполнится условие:
//     f(x) ≤ f(x - α ∇f(x)) - α * ε * ||∇f(x)||²
//   - После успешного выбора alpha обновляется точка: x := x - alpha * grad.
//
// Параметры:
// - alphaHat: начальное значение длины шага,
// - epsilon: параметр в правой части условия убывания,
// - lambda ∈ (0, 1): коэффициент уменьшения шага,
// - deltaGrad: порог нормы градиента.
//
// Особенности:
// - Гарантирует убывание функции на каждой итерации при достаточно малом alpha.
// - Сходится к стационарной точке при выполнении условий Липшица.
// - Не требует знания константы Липшица заранее.
//
// Возвращает xmin, ymin — координаты точки минимума,
// fmin — значение функции в этой точке, и iters — количество вызовов f.
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

// SteepestGradientDescent реализует метод наискорейшего градиентного спуска (МНГС)
// для минимизации функции двух переменных f(x, y).
//
// На каждой итерации выбирается направление антиградиента,
// и производится одномерная минимизация вдоль этого направления:
// α_k = argmin_{α ≥ 0} f(x_k - α ∇f(x_k))
// То есть шаг α подбирается оптимально по направлению антиградиента.
//
// Алгоритм:
// - Вычисляется градиент ∇f(x, y).
// - Если ||∇f|| ≤ gradEps, выполнение прекращается (достигнута стационарная точка).
// - Вдоль направления (-gx, -gy) строится функция φ(α) = f(x - αgx, y - αgy).
// - Параметр α минимизируется методом золотого сечения (на предварительно подобранном отрезке).
// - Точка обновляется: x ← x - α * gx, y ← y - α * gy.
//
// Особенности:
// - Обеспечивает наискорейшее уменьшение функции на каждом шаге (локально оптимальный шаг).
// - Шаг подбирается точно, а не приближённо (в отличие от градиентного метода с дроблением).
// - При выпуклости функции спуск стабильно сходится к минимуму.
// - Направления градиентов на соседних итерациях ортогональны (⟨∇f(x_k+1), ∇f(x_k)⟩ = 0).
//
// Возвращает: координаты точки минимума (xmin, ymin), значение функции в ней (fmin),
// и общее количество вызовов f (iters).
func SteepestGradientDescent(
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
		a, b := pkg.BracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x -= alpha * gx
		y -= alpha * gy
	}

	return x, y, phiF(x, y), iters
}

// AcceleratedGradientDescent реализует ускоренный градиентный метод p-го порядка
// для минимизации функции f(x, y) с использованием оптимизации вдоль направления
// между начальной и промежуточной точками градиентного спуска.
//
// Алгоритм состоит из двух этапов на каждой итерации:
// - Выполняется p шагов обычного градиентного спуска от текущей точки (x, y),
// получая промежуточную точку (xs, ys).
// - Производится одномерная минимизация по направлению от (x, y) к (xs, ys):
// ищется минимум φ(α) = f(x + α(dx), y + α(dy)) по α ≥ 0,
// где dx = xs - x, dy = ys - y.
// Это позволяет сделать "ускоряющий" шаг вдоль направления, проходящего вдоль "дна оврага".
//
// Таким образом, данный метод улучшает сходимость по сравнению с обычным методом
// наискорейшего градиентного спуска, особенно на сильно вытянутых (овражных) функциях.
//
// Параметры:
// - grad: функция, возвращающая градиент ∇f(x, y);
// - x0, y0: начальная точка;
// - p: число предварительных шагов наискорейшего спуска (рекомендуется p = dim);
// - gradEps: критерий остановы по норме градиента.
//
// Особенности:
// - Улучшает направление поиска за счёт приближённого выравнивания по овражной геометрии.
// - Требует выполнения (p + 1) одномерных оптимизаций на каждую итерацию.
// - Может достигать минимума быстрее, чем классический градиентный спуск.
//
// Возвращает: точку минимума (xmin, ymin), значение функции fmin и общее число вызовов функции (iters).
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
			a, b := pkg.BracketMinimum(phi1)
			alpha, _, _ := zeroordered.GoldenSectionSearch(phi1, a, b, gradEps)
			xs -= alpha * gxs
			ys -= alpha * gys
		}

		dx, dy := xs-x, ys-y
		phi2 := func(alpha float64) float64 {
			return phiF(x+alpha*dx, y+alpha*dy)
		}
		a, b := pkg.BracketMinimum(phi2)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi2, a, b, gradEps)

		x += alpha * dx
		y += alpha * dy
	}

	return x, y, phiF(x, y), iters
}

// RavineGradientDescent реализует овражный метод оптимизации для функции двух переменных f(x, y),
// предназначенный для ускорения сходимости на "овражных" функциях (сильно вытянутых в одном направлении).
//
// Идея метода:
//   - Из текущей точки x^k берется вторая точка x̃^k = x^k + δ вдоль диагонали (в обоих координатах).
//   - Из обеих точек (x^k и x̃^k) выполняется p шагов градиентного спуска —
//     это даёт две точки y^k и ỹ^k, приблизительно лежащие на "дне оврага".
//   - Затем проводится одномерная минимизация по направлению между этими двумя точками (ỹ^k - y^k),
//     начиная от y^k, чтобы получить новую точку x^{k+1}.
//
// Параметры:
// - grad: функция градиента ∇f(x, y);
// - x0, y0: начальная точка;
// - delta: смещение для второй стартовой точки x̃^k (обычно малое);
// - p: число шагов градиентного спуска от x^k и x̃^k;
// - gradEps: критерий остановы по норме градиента.
//
// Особенности:
//   - Эффективен при оптимизации овражных функций, где стандартный градиентный спуск
//     "застревает" в зигзагообразной траектории вдоль крутого оврага.
//   - Вычислительно дороже, чем обычный градиентный спуск: требует 2×p градиентных шагов и одну линейную оптимизацию на итерацию.
//   - Позволяет быстро сойтись вдоль направления минимального изменения.
//
// Возвращает: координаты минимума (xmin, ymin), значение функции в этой точке (fmin),
// и общее число вызовов функции (iters).
func RavineGradientDescent(
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
			a, b := pkg.BracketMinimum(phi)
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
			a, b := pkg.BracketMinimum(phi)
			alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)
			xst -= alpha * gxst
			yst -= alpha * gyst
		}

		dx, dy := xst-xs, yst-ys

		phi := func(alpha float64) float64 {
			return phiF(xs+alpha*dx, ys+alpha*dy)
		}
		a, b := pkg.BracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x = xs + alpha*dx
		y = ys + alpha*dy
	}

	return x, y, phiF(x, y), iters
}

// NewtonModified реализует модифицированный метод Ньютона для двумерной функции f(x, y),
// комбинируя второй порядок метода Ньютона с одномерной оптимизацией шага α.
// В отличие от классического метода, здесь α подбирается точно (через одномерную минимизацию),
// что повышает стабильность сходимости при плохо всё ещё заданном начальном приближении.
//
// Алгоритм:
// - Вычислить градиент ∇f = (gx, gy). Если его норма ≤ gradEps — остановить вычисления.
// - Построить матрицу Гессе H = [[hxx, hxy], [hyx, hyy]] и проверить её невырожденность.
// - Найти направление спуска p = –H⁻¹ ∇f через решение 2×2 системы:
// det = hxx*hyy – hxy*hyx
// p = (px, py) = –(H⁻¹ ∇f).
// - Задайте функцию φ(α) = f( x + α·px, y + α·py ) и подберите оптимальный α ≥ 0
// на отрезке [a, b], содержащем минимум.
// - Обновить точку: (x, y) ← (x, y) + α · p и повторить.
//
// Параметры:
// - f: целевая функция;
// - grad: функция, возвращающая (∂f/∂x, ∂f/∂y);
// - hess: функция, возвращающая элементы Гессиана (hxx, hxy, hyx, hyy);
// - x0, y0: начальная точка;
// - gradEps: порог по норме градиента для остановы.
//
// Особенности:
// - Квадратичная сходимость при окрестности решения и невырожденном Гессиане.
// - Вычисляется H⁻¹ ∇f явно для 2D — простой аналитический разворот матрицы.
// - Точный шаг α из одномерной оптимизации улучшает глобальную сходимость.
// - Каждая итерация делает одну двумерную обратную матрицу + одну одномерную оптимизацию.
//
// Возвращает координаты xmin, ymin — найденного минимума, fmin — значение f в этой точке,
// и iters — число вызовов f (для оценки вычислительной стоимости).
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
		a, b := pkg.BracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x += alpha * px
		y += alpha * py
	}

	return x, y, phiF(x, y), iters
}

// QuasiNewton реализует двумерный квази-ньютоновский метод с поправкой ранга 1 (rank-1 update)
// для минимизации функции f(x, y). Вместо вычисления и обращения Гессиана на каждой итерации
// строится его аппроксимация H_k, которая обновляется по формуле:
// H_{k+1} = H_k + (δ_k − H_k γ_k)(δ_k − H_k γ_k)^T / ((δ_k − H_k γ_k)^T γ_k),
// где δ_k = x_{k+1} − x_k, γ_k = ∇f(x_{k+1}) − ∇f(x_k).
//
// На каждой итерации:
//  1. Вычисляем градиент ∇f = (gx, gy). Если ||∇f|| ≤ gradEps — останавливаемся.
//  2. Задаём направление p = –H_k ∇f.
//  3. Оптимизируем шаг α ≥ 0 вдоль p.
//  4. Обновляем точку: x ← x + α p.
//  5. Считаем δ = x_{new} − x_old, γ = ∇f_{new} − ∇f_old.
//  6. Вычисляем rank-1 поправку v = δ − H_k γ, и если v^T γ ≠ 0, добавляем её в H_k.
//  7. После каждых n = 2 итераций (k%2==0) сбрасываем H_k на единичную матрицу,
//     чтобы сохранить симметрию и избежать накопления ошибок.
//
// Параметры:
// - f: функция двух переменных.
// - grad: функция, возвращающая её градиент (gx, gy).
// - x0, y0: начальное приближение.
// - gradEps: порог по норме градиента для остановы.
//
// Возвращает:
// - xmin, ymin: найденная точка минимума,
// - fmin: значение f в ней,
// - iters: число вызовов f (оценка стоимости вычислений).
func QuasiNewton(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	Hxx, Hxy := 1.0, 0.0
	Hyx, Hyy := 0.0, 1.0

	var k int
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	for {
		k++
		gx, gy := grad(x, y)
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		px := -(Hxx*gx + Hxy*gy)
		py := -(Hyx*gx + Hyy*gy)

		phi := func(alpha float64) float64 {
			return phiF(x+alpha*px, y+alpha*py)
		}
		a, b := pkg.BracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		xNew, yNew := x+alpha*px, y+alpha*py
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
		if k%2 == 0 {
			Hxx, Hxy = 1.0, 0.0
			Hyx, Hyy = 0.0, 1.0
		}

		x, y = xNew, yNew
	}

	return x, y, phiF(x, y), iters
}

// ConjGradFR реализует метод сопряжённых направлений Флетчера–Ривза
// для двумерной минимизации функции f(x, y). Периодически (каждые 2 шага)
// направление сбрасывается на чистый антиградиент, чтобы восстановить
// сопряжённость и ограничить накопление погрешностей.
//
// Алгоритм:
//  1. Инициализируем x₀ = (x0, y0), вычисляем g₀ = ∇f(x₀), d₀ = −g₀.
//  2. Для k = 0, 1, 2, ... до сходимости:
//     a) Если ‖gₖ‖ ≤ gradEps — выходим (достигли стационарной точки).
//     b) Минимизируем вдоль dₖ: φ(α)=f(xₖ+α dₖ) → min, находим αₖ ≥ 0.
//     c) Обновляем xₖ₊1 = xₖ + αₖ dₖ.
//     d) Вычисляем gₖ₊1 = ∇f(xₖ₊1).
//     e) Вычисляем βₖ = ‖gₖ₊1‖² / ‖gₖ‖² (если знаменатель > 0).
//     f) Если (k+1)%2 == 0, то dₖ₊1 = −gₖ₊1
//     иначе dₖ₊1 = −gₖ₊1 + βₖ dₖ.
//  3. Повторяем, пока не выполнится условие остановы.
//
// Параметры:
// - f: функция двух переменных.
// - grad: возвращает её градиент (gx, gy).
// - x0, y0: начальное приближение.
// - gradEps: порог по норме градиента.
//
// Возвращает:
// - xmin, ymin: найденную точку минимума.
// - fmin: значение f в xmin,ymin.
// - iters: число вызовов f.
func ConjGradFR(
	f func(x, y float64) float64,
	grad func(x, y float64) (gx, gy float64),
	x0, y0, gradEps float64,
) (xmin, ymin, fmin float64, iters int) {
	var k int
	// текущее приближение
	x, y := x0, y0

	phiF := func(x_, y_ float64) float64 {
		iters++
		return f(x_, y_)
	}

	gx, gy := grad(x, y)
	dx, dy := -gx, -gy

	for {
		k++
		if math.Hypot(gx, gy) <= gradEps {
			break
		}

		phi := func(alpha float64) float64 {
			return phiF(x+alpha*dx, y+alpha*dy)
		}
		a, b := pkg.BracketMinimum(phi)
		alpha, _, _ := zeroordered.GoldenSectionSearch(phi, a, b, gradEps)

		x += alpha * dx
		y += alpha * dy

		gxNew, gyNew := grad(x, y)

		// Метод Флетчера-Ривза: β_k = (‖g_{k+1}‖²)/(‖g_k‖²)
		num := gxNew*gxNew + gyNew*gyNew
		den := gx*gx + gy*gy

		var beta float64
		if den > 0 {
			beta = num / den
		}

		// сброс направлений каждые n=2 шага
		if k%2 == 0 {
			dx, dy = -gxNew, -gyNew
		} else {
			dx = -gxNew + beta*dx
			dy = -gyNew + beta*dy
		}

		gx, gy = gxNew, gyNew
	}

	xmin, ymin = x, y
	fmin = phiF(x, y)

	return xmin, ymin, fmin, iters
}
