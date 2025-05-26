package zeroordered

import (
	"math"
	"testing"
)

func BenchmarkFibonacciSearch(b *testing.B) {
	testFunc := func(x float64) float64 {
		return math.Pow(x-2, 2)
	}
	a, c, eps := 0.0, 4.0, 1e-5

	for i := 0; i < b.N; i++ {
		FibonacciSearch(testFunc, a, c, eps)
	}
}

func BenchmarkGoldenSectionSearch(b *testing.B) {
	testFunc := func(x float64) float64 {
		return math.Pow(x-2, 2)
	}
	a, c, eps := 0.0, 4.0, 1e-5

	for i := 0; i < b.N; i++ {
		GoldenSectionSearch(testFunc, a, c, eps)
	}
}

func TestPassiveSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}

	xmin, fmin, iterations := PassiveSearch(f, a, b, eps)

	if xmin != 1.5 {
		t.Errorf("xmin = %v, expected 1.5", xmin)
	}

	if fmin-2.83 > eps {
		t.Errorf("fmin = %v, expected 2.83", fmin)
	}

	if iterations != 7 {
		t.Errorf("iterations = %v, expected 7", iterations)
	}
}

func TestDichotomySearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	delta := 0.1
	f := func(x float64) float64 {
		return x + 2/x
	}
	xmin, fmin, aFinal, bFinal, iterations := DichotomySearch(f, a, b, eps, delta)
	if xmin-1.638 > eps {
		t.Errorf("xmin = %v, expected 1.638", xmin)
	}
	if fmin-2.86 > eps {
		t.Errorf("fmin = %v, expected 2.86", fmin)
	}
	if aFinal-1.225 > eps {
		t.Errorf("aFinal = %v, expected 1.225", aFinal)
	}
	if bFinal-2.05 > eps {
		t.Errorf("bFinal = %v, expected 2.05", bFinal)
	}
	if iterations != 5 {
		t.Errorf("iterations = %v, expected 4", iterations)
	}
}

func TestGoldenSectionSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}
	xmin, fmin, i := GoldenSectionSearch(f, a, b, eps)
	if xmin-1.562 > eps {
		t.Errorf("xmin = %v, expected 1.562", xmin)
	}
	if fmin-2.84 > eps {
		t.Errorf("fmin = %v, expected 2.84", fmin)
	}
	if i != 5 {
		t.Errorf("i = %v, expected 5", i)
	}
}

func TestFibonacciSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}
	xmin, fmin, i := FibonacciSearch(f, a, b, eps)
	if xmin-1.25 > eps {
		t.Errorf("xmin = %v, expected 1.25", xmin)
	}
	if fmin-2.85 > eps {
		t.Errorf("fmin = %v, expected 2.85", fmin)
	}
	if i != 6 {
		t.Errorf("i = %v, expected 6", i)
	}
}
