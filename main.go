package main

import (
	"fmt"

	zeroordered "github.com/vshulcz/edu_optimization_methods/internal/1_zero_ordered"
	highordered "github.com/vshulcz/edu_optimization_methods/internal/2_high_ordered"
	multidimensional "github.com/vshulcz/edu_optimization_methods/internal/3_multidimensional"
	conditional "github.com/vshulcz/edu_optimization_methods/internal/4_conditional"
	"github.com/vshulcz/edu_optimization_methods/pkg"
)

const (
	a       = 1
	b       = 2
	epsilon = 0.02
	delta   = 0.001
)

func main() {
	xmin, fmin, iterations := zeroordered.PassiveSearch(pkg.F1, a, b, epsilon)
	fmt.Printf("Метод пассивного поиска:\n")
	fmt.Printf("Минимум функции: x = %v, f(x) = %v\n", xmin, fmin)
	fmt.Printf("Гарантированный интервал для истинного минимума: [%f, %f]\n", xmin-epsilon, xmin+epsilon)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, aFinal, bFinal, iterations := zeroordered.DichotomySearch(pkg.F1, a, b, epsilon, delta)
	fmt.Printf("Метод дихотомии:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Локализующий интервал: [%f, %f]\n", aFinal, bFinal)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = zeroordered.GoldenSectionSearch(pkg.F1, a, b, epsilon)
	fmt.Printf("Метод золотого сечения:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = zeroordered.FibonacciSearch(pkg.F1, a, b, epsilon)
	fmt.Printf("Метод Фибоначчи:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = highordered.TangentSearch(pkg.F1, pkg.DF1, a, b, epsilon)
	fmt.Printf("Метод касательных:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = highordered.NewtonSearch(pkg.F1, pkg.DF1, pkg.DDF1, a, epsilon)
	fmt.Printf("Метод Ньютона-Рафсона:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = highordered.SecantSearch(pkg.F1, pkg.DF1, a, b, epsilon)
	fmt.Printf("Метод секущих:\n")
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations := multidimensional.CoordinateDescent(pkg.F2, 1, 1, -4, 4, -4, 4, epsilon)
	fmt.Printf("Метод покоординатного спуска:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.GradientDescentBacktracking(pkg.F2, pkg.GradF2, 0, 0, 1.0, epsilon, 0.5, 1e-4)
	fmt.Printf("Градиентный метод с дроблением шага:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.SteepestGradientDescent(pkg.F2, pkg.GradF2, 0, 0, epsilon)
	fmt.Printf("Метод наискорейшего градиентного спуска:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.AcceleratedGradientDescent(pkg.F2, pkg.GradF2, 0, 0, 2, epsilon)
	fmt.Printf("Ускоренный градиентный метод p-го порядка:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.RavineGradientDescent(pkg.F2, pkg.GradF2, 0, 0, 0.5, 1, epsilon)
	fmt.Printf("Овражный метод:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.NewtonModified(pkg.F2, pkg.GradF2, pkg.HessF2, 0, 0, epsilon)
	fmt.Printf("Модифицированный метод Ньютона:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.QuasiNewton(pkg.F2, pkg.GradF2, 0, 0, epsilon)
	fmt.Printf("Квазиньютоновский метод:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, iterations = multidimensional.ConjGradFR(pkg.F2, pkg.GradF2, 0, 0, epsilon)
	fmt.Printf("Метод сопряженных отрезков:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, l1, l2, iterations := conditional.KuhnTucker(pkg.F3, pkg.GradF3, epsilon)
	fmt.Printf("Метод Куна-Таккера:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f\n", xmin, ymin, fmin)
	fmt.Printf("Найденные лямбды %f, %f\n", l1, l2)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, ymin, fmin, r, outerIt := conditional.ExternalPenalty(
		pkg.F3, pkg.GradF3,
		0, 0,
		1,
		10,
		0.01,
		0.001,
		10,
	)
	fmt.Printf("Метод Внешних штрафов:\n")
	fmt.Printf("Минимум найден в точке (x,y) = (%f, %f), f(x,y) = %f, (r=%f)\n", xmin, ymin, fmin, r)
	fmt.Printf("Количество итераций: %d\n\n", outerIt)
}
