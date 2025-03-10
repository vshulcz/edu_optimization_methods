package method1

import (
	"testing"
)

func TestPassiveSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}

	xmin, fmin, iterations := PassiveSearch(f, a, b, eps)

	if xmin != 1.5 {
		t.Errorf("xmin = %v, ожидается 1.5", xmin)
	}

	if fmin-2.83 > eps {
		t.Errorf("fmin = %v, ожидается 2.83", fmin)
	}

	if iterations != 7 {
		t.Errorf("iterations = %v, ожидается 7", iterations)
	}
}
