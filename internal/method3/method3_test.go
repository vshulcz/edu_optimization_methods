package method3

import (
	"testing"
)

func TestGoldenSectionSearch(t *testing.T) {
	a := 0.5
	b := 3.5
	eps := 0.5
	f := func(x float64) float64 {
		return x + 2/x
	}
	xmin, fmin, i := GoldenSectionSearch(f, a, b, eps)
	if xmin-1.562 > eps {
		t.Errorf("xmin = %v, ожидается 1.562", xmin)
	}
	if fmin-2.84 > eps {
		t.Errorf("fmin = %v, ожидается 2.84", fmin)
	}
	if i != 5 {
		t.Errorf("i = %v, ожидается 5", i)
	}
}
