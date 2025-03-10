package method1

import (
	"math"
)

func PassiveSearch(f func(x float64) float64, a, b, eps float64) (xmin, fmin float64, iterations int) {
	k := int(math.Ceil((b - a) / eps))

	minVal := math.Inf(1)
	minX := a

	for i := 0; i <= k; i++ {
		x := a + float64(i)*(b-a)/float64(k)
		val := f(x)
		if val < minVal {
			minVal = val
			minX = x
		}
		iterations++
	}

	return minX, minVal, iterations
}
