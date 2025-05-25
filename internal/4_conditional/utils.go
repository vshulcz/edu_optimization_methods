package conditional

import "errors"

func SolveGauss(A []float64, b []float64, n int) ([]float64, error) {
	M := make([]float64, n*(n+1))
	for i := 0; i < n; i++ {
		copy(M[i*(n+1):], A[i*n:(i+1)*n])
		M[i*(n+1)+n] = b[i]
	}

	const eps = 1e-12
	for i := 0; i < n; i++ {
		pivot := i
		for j := i + 1; j < n; j++ {
			if abs(M[j*(n+1)+i]) > abs(M[pivot*(n+1)+i]) {
				pivot = j
			}
		}
		if abs(M[pivot*(n+1)+i]) < eps {
			return nil, errors.New("matrix is singular")
		}
		if pivot != i {
			for k := i; k < n+1; k++ {
				M[i*(n+1)+k], M[pivot*(n+1)+k] =
					M[pivot*(n+1)+k], M[i*(n+1)+k]
			}
		}
		diag := M[i*(n+1)+i]
		for k := i; k < n+1; k++ {
			M[i*(n+1)+k] /= diag
		}
		for u := 0; u < n; u++ {
			if u == i {
				continue
			}
			factor := M[u*(n+1)+i]
			for k := i; k < n+1; k++ {
				M[u*(n+1)+k] -= factor * M[i*(n+1)+k]
			}
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = M[i*(n+1)+n]
	}
	return x, nil
}

func abs(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}
