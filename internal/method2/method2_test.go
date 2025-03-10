package method2

import (
	"testing"
)

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
		t.Errorf("xmin = %v, ожидается 1.638", xmin)
	}
	if fmin-2.86 > eps {
		t.Errorf("fmin = %v, ожидается 2.86", fmin)
	}
	if aFinal-1.225 > eps {
		t.Errorf("aFinal = %v, ожидается 1.225", aFinal)
	}
	if bFinal-2.05 > eps {
		t.Errorf("bFinal = %v, ожидается 2.05", bFinal)
	}
	if iterations != 4 {
		t.Errorf("iterations = %v, ожидается 4", iterations)
	}
}
