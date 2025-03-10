package main

import (
	"fmt"

	"github.com/vshulcz/edu_optimization_methods/internal/method1"
	"github.com/vshulcz/edu_optimization_methods/internal/method2"
	"github.com/vshulcz/edu_optimization_methods/internal/method3"
	"github.com/vshulcz/edu_optimization_methods/internal/method4"
	"github.com/vshulcz/edu_optimization_methods/pkg"
)

const (
	a       = 1
	b       = 2
	epsilon = 0.02
	delta   = 0.001
)

func main() {
	xmin, fmin, iterations := method1.PassiveSearch(pkg.F, a, b, epsilon)
	fmt.Printf("Минимум функции: x = %v, f(x) = %v\n", xmin, fmin)
	fmt.Printf("Гарантированный интервал для истинного минимума: [%f, %f]\n", xmin-epsilon, xmin+epsilon)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, aFinal, bFinal, iterations := method2.DichotomySearch(pkg.F, a, b, epsilon, delta)
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Локализующий интервал: [%f, %f]\n", aFinal, bFinal)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = method3.GoldenSectionSearch(pkg.F, a, b, epsilon)
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n\n", iterations)

	xmin, fmin, iterations = method4.FibonacciSearch(pkg.F, a, b, epsilon)
	fmt.Printf("Минимум найден в точке x = %f, f(x) = %f\n", xmin, fmin)
	fmt.Printf("Количество итераций: %d\n", iterations)
}
