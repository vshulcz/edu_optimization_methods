package method4

import (
	"testing"
)

func TestFibonacciSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}
	xmin, fmin, i := FibonacciSearch(f, a, b, eps)
	if xmin-1.25 > eps {
		t.Errorf("xmin = %v, ожидается 1.25", xmin)
	}
	if fmin-2.85 > eps {
		t.Errorf("fmin = %v, ожидается 2.85", fmin)
	}
	if i != 4 {
		t.Errorf("i = %v, ожидается 4", i)
	}
}
