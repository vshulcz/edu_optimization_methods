package highordered

import (
	"math"
	"testing"
)

func TestTangentSearch(t *testing.T) {
	type args struct {
		f   func(x float64) float64
		df  func(x float64) float64
		a   float64
		b   float64
		eps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "f(x) = x^2 + 2x + 1 (minimum at x = -1)",
			args: args{
				f:   func(x float64) float64 { return x*x + 2*x + 1 },
				df:  func(x float64) float64 { return 2*x + 2 },
				a:   -3.0,
				b:   2.0,
				eps: 1e-6,
			},
			wantXmin:  -1.0,
			wantFmin:  0.0,
			wantIters: 43,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotFmin, gotIters := TangentSearch(tt.args.f, tt.args.df, tt.args.a, tt.args.b, tt.args.eps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("TangentSearch() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("TangentSearch() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("TangentSearch() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestNewtonSearch(t *testing.T) {
	type args struct {
		f   func(x float64) float64
		df  func(x float64) float64
		d2f func(x float64) float64
		x0  float64
		eps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x) = x + 2/x",
			args: args{
				f:   func(x float64) float64 { return x + 2/x },
				df:  func(x float64) float64 { return 1 - 2/(x*x) },
				d2f: func(x float64) float64 { return 4 / (x * x * x) },
				x0:  0.5,
				eps: 0.5,
			},
			wantXmin:  1.239,
			wantFmin:  2.853,
			wantIters: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotFmin, gotIters := NewtonSearch(tt.args.f, tt.args.df, tt.args.d2f, tt.args.x0, tt.args.eps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-3 {
				t.Errorf("NewtonSearch() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-3 {
				t.Errorf("NewtonSearch() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("NewtonSearch() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestSecantSearch(t *testing.T) {
	type args struct {
		f   func(x float64) float64
		df  func(x float64) float64
		x0  float64
		x1  float64
		eps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x) = (x-2)^2",
			args: args{
				f:   func(x float64) float64 { return (x - 2) * (x - 2) },
				df:  func(x float64) float64 { return 2 * (x - 2) },
				x0:  0.0,
				x1:  5.0,
				eps: 1e-6,
			},
			wantXmin:  2.0,
			wantFmin:  0,
			wantIters: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotFmin, gotIters := SecantSearch(tt.args.f, tt.args.df, tt.args.x0, tt.args.x1, tt.args.eps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-3 {
				t.Errorf("SecantSearch() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-3 {
				t.Errorf("SecantSearch() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("SecantSearch() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}
