package highordered

import (
	"math"
)

// TangentSearch реализует метод касательных (метод хорд) для поиска минимума функции f
// на отрезке [a, b] с использованием её производной df и заданной точностью eps.
//
// Метод основан на построении касательных к графику функции в точках a и b,
// затем на нахождении их точки пересечения, которая используется для уточнения локализации минимума.
//
// Алгоритм:
//   - Если df(a) ≥ 0, то минимум находится в точке a.
//   - Если df(b) ≤ 0, то минимум в точке b.
//   - Иначе итеративно строятся касательные в a и b, ищется точка их пересечения x0:
//     x0 = (c2 - c1) / (m1 - m2), где c1 = f(a) - m1*a, c2 = f(b) - m2*b
//   - Вычисляется df(x0), и в зависимости от его знака отрезок [a, b] сужается:
//     если df(x0) > 0, то b = x0; иначе a = x0.
//   - Процесс повторяется до выполнения одного из условий остановы:
//     |b - a| ≤ eps или |df(x0)| ≤ eps
//
// Особенности:
// - Использует первую производную функции — метод первого порядка.
// - Эффективен для выпуклых и гладких функций, где f'(a) < 0 и f'(b) > 0.
// - Быстрая сходимость при хорошо заданных начальных условиях.
//
// Возвращает xmin — найденную стационарную точку, fmin — значение функции в ней,
// и iters — количество вызовов f.
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

// NewtonSearch реализует метод Ньютона-Рафсона (Newton-Raphson method)
// для поиска точки минимума функции f, решая уравнение f'(x) = 0
// с использованием первой и второй производных df и d2f соответственно.
//
// Алгоритм итеративно уточняет приближение x с помощью формулы:
// x_{k+1} = x_k - f'(x_k) / f”(x_k)
//
// То есть на каждой итерации проводится касательная к f'(x),
// и точка пересечения касательной с осью X становится следующим приближением.
//
// Условия остановы:
// - |f'(x)| ≤ eps — достигнута стационарная точка (точность по производной)
// - f”(x) = 0 — невозможность деления, итерации прекращаются
//
// Особенности:
// - Метод второго порядка: при хороших начальных условиях сходится квадратично.
// - Требует знание второй производной функции.
// - Не гарантирует сходимость, особенно при плохом начальном приближении или если f”(x) близко к нулю.
// - Подходит для задач, где f(x) дважды дифференцируема и минимум — стационарная точка.
//
// Возвращает xmin — точку, где f'(x) ≈ 0, fmin — значение функции в ней,
// и iters — количество вызовов f.
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

// SecantSearch реализует метод секущих (или метод хорд) для поиска стационарной точки,
// т.е. решения уравнения f'(x) = 0, основываясь только на первой производной df.
// Метод не требует второй производной, в отличие от метода Ньютона.
//
// Алгоритм использует аппроксимацию второй производной с помощью разностного отношения:
// f”(x_k) ≈ (f'(x_k) - f'(x_{k-1})) / (x_k - x_{k-1})
//
// Это приводит к итерационной формуле:
// x_{k+1} = x_k - (x_k - x_{k-1}) * f'(x_k) / (f'(x_k) - f'(x_{k-1}))
//
// Алгоритм:
//   - Начинается с двух начальных приближений x0 и x1.
//   - На каждой итерации вычисляется новое приближение x2 как точка пересечения секущей
//     между (x_k, f'(x_k)) и (x_{k-1}, f'(x_{k-1})) с осью X.
//   - Процесс повторяется, пока |f'(x_k)| ≤ eps или знаменатель не станет нулевым.
//
// Особенности:
// - Метод первого порядка (медленнее Ньютона, но не требует f”).
// - Чувствителен к выбору начальных приближений.
// - Может расходиться или зациклиться при плохом выборе x0, x1.
//
// Возвращает xmin — приближение к стационарной точке, fmin — значение функции в ней,
// и iters — количество вызовов функции f.
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
