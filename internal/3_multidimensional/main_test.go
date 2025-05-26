package multidimensional

import (
	"math"
	"testing"
)

func TestCoordinateDescent(t *testing.T) {
	type args struct {
		f   func(x, y float64) float64
		x0  float64
		y0  float64
		ax  float64
		bx  float64
		ay  float64
		by  float64
		eps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				x0:  1.0,
				y0:  1.0,
				ax:  -4.0,
				bx:  4.0,
				ay:  -4.0,
				by:  4.0,
				eps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 428,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := CoordinateDescent(tt.args.f, tt.args.x0, tt.args.y0, tt.args.ax, tt.args.bx, tt.args.ay, tt.args.by, tt.args.eps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-4 {
				t.Errorf("CoordinateDescent() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-4 {
				t.Errorf("CoordinateDescent() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-4 {
				t.Errorf("CoordinateDescent() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("CoordinateDescent() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestGradientDescentBacktracking(t *testing.T) {
	type args struct {
		f         func(x, y float64) float64
		grad      func(x, y float64) (gx, gy float64)
		x0        float64
		y0        float64
		alphaHat  float64
		epsilon   float64
		lambda    float64
		deltaGrad float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:        1.0,
				y0:        1.0,
				alphaHat:  0.1,
				epsilon:   1e-6,
				lambda:    0.1,
				deltaGrad: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 42,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := GradientDescentBacktracking(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.alphaHat, tt.args.epsilon, tt.args.lambda, tt.args.deltaGrad)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("GradientDescentBacktracking() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("GradientDescentBacktracking() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("GradientDescentBacktracking() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("GradientDescentBacktracking() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestSteepestGradientDescent(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		x0      float64
		y0      float64
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 166,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 0.05,
			},
			wantXmin:  0.0009394268730375316,
			wantYmin:  0.01440979491160928,
			wantFmin:  0.00021558489504270635,
			wantIters: 141,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := SteepestGradientDescent(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("SteepestGradientDescent() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("SteepestGradientDescent() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("SteepestGradientDescent() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("SteepestGradientDescent() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestAcceleratedGradientDescent(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		x0      float64
		y0      float64
		p       int
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:      1.0,
				y0:      1.0,
				p:       2,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 205,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				x0:      1.0,
				y0:      1.0,
				p:       2,
				gradEps: 0.05,
			},
			wantXmin:  -0.0001747128435803141,
			wantYmin:  -0.0034917214256659033,
			wantFmin:  1.2466839713861601e-05,
			wantIters: 103,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := AcceleratedGradientDescent(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.p, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("AcceleratedGradientDescent() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("AcceleratedGradientDescent() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("AcceleratedGradientDescent() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("AcceleratedGradientDescent() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestRavineGradientDescent(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		x0      float64
		y0      float64
		delta   float64
		p       int
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:      1.0,
				y0:      1.0,
				p:       2,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 496,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				x0:      1.0,
				y0:      1.0,
				p:       2,
				gradEps: 0.05,
			},
			wantXmin:  0.0009394268730375316,
			wantYmin:  0.0144097949116092,
			wantFmin:  0.00021558489504270635,
			wantIters: 351,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := RavineGradientDescent(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.delta, tt.args.p, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("RavineGradientDescent() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("RavineGradientDescent() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("RavineGradientDescent() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("RavineGradientDescent() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestNewtonModified(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		hess    func(x, y float64) (hxx, hxy, hyx, hyy float64)
		x0      float64
		y0      float64
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				hess: func(x, y float64) (hxx, hxy, hyx, hyy float64) {
					E := math.Exp(x*x + y*y)
					hxx = 2 + 2*E + 4*x*x*E
					hyy = 2*E + 4*y*y*E
					hxy = 4 * x * y * E
					hyx = hxy
					return
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 188,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				hess: func(x, y float64) (hxx, hxy, hyx, hyy float64) {
					hxx = 18
					hxy = 0
					hyx = 0
					hyy = 2
					return
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 0.05,
			},
			wantXmin:  0.0004531268730375316,
			wantYmin:  0.000453197949116092,
			wantFmin:  0.0000021558489504270,
			wantIters: 29,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := NewtonModified(tt.args.f, tt.args.grad, tt.args.hess, tt.args.x0, tt.args.y0, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("NewtonModified() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("NewtonModified() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("NewtonModified() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("NewtonModified() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestQuasiNewton(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		x0      float64
		y0      float64
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 166,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 0.05,
			},
			wantXmin:  0.0016676491865052516,
			wantYmin:  0.0022439658770094474,
			wantFmin:  3.0064867140447425e-05,
			wantIters: 51,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := QuasiNewton(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("QuasiNewton() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("QuasiNewton() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("QuasiNewton() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("QuasiNewton() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func TestConjGradFR(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		x0      float64
		y0      float64
		gradEps float64
	}
	tests := []struct {
		name      string
		args      args
		wantXmin  float64
		wantYmin  float64
		wantFmin  float64
		wantIters int
	}{
		{
			name: "Case 1: f(x,y) = x*x + math.Exp(x*x+y*y) + 4*x + 3*y",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + math.Exp(x*x+y*y) + 4*x + 3*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2*x + 2*x*math.Exp(x*x+y*y) + 4, 2*y*math.Exp(x*x+y*y) + 3
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 1e-6,
			},
			wantXmin:  -0.613225,
			wantYmin:  -0.663293,
			wantFmin:  -1.805292,
			wantIters: 166,
		},
		{
			name: "Case 2: f(x,y) = 9*x*x + y*y",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18 * x, 2 * y
				},
				x0:      1.0,
				y0:      1.0,
				gradEps: 0.05,
			},
			wantXmin:  0.0004625978729986543,
			wantYmin:  0.0023294319568673144,
			wantFmin:  7.352224370600598e-06,
			wantIters: 61,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotIters := ConjGradFR(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.gradEps)
			if math.Abs(gotXmin-tt.wantXmin) > 1e-6 {
				t.Errorf("ConjGradFR() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if math.Abs(gotYmin-tt.wantYmin) > 1e-6 {
				t.Errorf("ConjGradFR() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if math.Abs(gotFmin-tt.wantFmin) > 1e-6 {
				t.Errorf("ConjGradFR() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if gotIters != tt.wantIters {
				t.Errorf("ConjGradFR() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}
