package conditional

import (
	"math"
	"testing"
)

func TestKuhnTucker(t *testing.T) {
	type args struct {
		f       func(x, y float64) float64
		grad    func(x, y float64) (gx, gy float64)
		gradEps float64
	}
	tests := []struct {
		name        string
		args        args
		wantXmin    float64
		wantYmin    float64
		wantFmin    float64
		wantLam1Opt float64
		wantLam2Opt float64
		wantIters   int
	}{
		{
			name: "F: 9x²+y²-54x+4y on x>=0,y>=0",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y - 54*x + 4*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18*x - 54, 2*y + 4
				},
				gradEps: 1e-6,
			},
			wantXmin:    3.0,
			wantYmin:    0.0,
			wantFmin:    -81.0,
			wantLam1Opt: 0.0,
			wantLam2Opt: 4.0,
			wantIters:   40,
		},
		{
			name: "F: x²+y² on x>=0,y>=0",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2 * x, 2 * y
				},
				gradEps: 1e-6,
			},
			wantXmin:    0.0,
			wantYmin:    0.0,
			wantFmin:    0.0,
			wantLam1Opt: 0.0,
			wantLam2Opt: 0.0,
			wantIters:   1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotLam1Opt, gotLam2Opt, gotIters := KuhnTucker(tt.args.f, tt.args.grad, tt.args.gradEps)
			if !equal(gotXmin, tt.wantXmin, tt.args.gradEps) {
				t.Errorf("KuhnTucker() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if !equal(gotYmin, tt.wantYmin, tt.args.gradEps) {
				t.Errorf("KuhnTucker() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if !equal(gotFmin, tt.wantFmin, tt.args.gradEps) {
				t.Errorf("KuhnTucker() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if !equal(gotLam1Opt, tt.wantLam1Opt, tt.args.gradEps) {
				t.Errorf("KuhnTucker() gotLam1Opt = %v, want %v", gotLam1Opt, tt.wantLam1Opt)
			}
			if !equal(gotLam2Opt, tt.wantLam2Opt, tt.args.gradEps) {
				t.Errorf("KuhnTucker() gotLam2Opt = %v, want %v", gotLam2Opt, tt.wantLam2Opt)
			}
			if gotIters != tt.wantIters {
				t.Errorf("KuhnTucker() gotIters = %v, want %v", gotIters, tt.wantIters)
			}
		})
	}
}

func equal(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestExternalPenalty(t *testing.T) {
	type args struct {
		f            func(x, y float64) float64
		grad         func(x, y float64) (gx, gy float64)
		x0           float64
		y0           float64
		r0           float64
		rFactor      float64
		epsConstr    float64
		epsGrad      float64
		alpha        float64
		maxOuterIter int
	}
	tests := []struct {
		name        string
		args        args
		wantXmin    float64
		wantYmin    float64
		wantFmin    float64
		wantR       float64
		wantOuterIt int
	}{
		{
			name: "F: 9x²+y²-54x+4y on x>=0,y>=0",
			args: args{
				f: func(x, y float64) float64 {
					return 9*x*x + y*y - 54*x + 4*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 18*x - 54, 2*y + 4
				},
				x0:           0,
				y0:           0,
				r0:           1,
				rFactor:      10,
				epsConstr:    0.01,
				epsGrad:      0.001,
				alpha:        0.01,
				maxOuterIter: 10,
			},
			wantXmin:    3.0,
			wantYmin:    -0.019804196101694333,
			wantFmin:    -81.07882457821823,
			wantR:       100.0,
			wantOuterIt: 3,
		},
		{
			name: "F: x²+y² on x>=0,y>=0",
			args: args{
				f: func(x, y float64) float64 {
					return x*x + y*y
				},
				grad: func(x, y float64) (gx, gy float64) {
					return 2 * x, 2 * y
				},
				x0:           0,
				y0:           0,
				r0:           1,
				rFactor:      10,
				epsConstr:    0.01,
				epsGrad:      0.001,
				alpha:        0.01,
				maxOuterIter: 10,
			},
			wantXmin:    0.0,
			wantYmin:    0.0,
			wantFmin:    0.0,
			wantR:       1.0,
			wantOuterIt: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotXmin, gotYmin, gotFmin, gotR, gotOuterIt := ExternalPenalty(tt.args.f, tt.args.grad, tt.args.x0, tt.args.y0, tt.args.r0, tt.args.rFactor, tt.args.epsConstr, tt.args.epsGrad, tt.args.maxOuterIter)
			if !equal(gotXmin, tt.wantXmin, tt.args.epsGrad) {
				t.Errorf("ExternalPenalty() gotXmin = %v, want %v", gotXmin, tt.wantXmin)
			}
			if !equal(gotYmin, tt.wantYmin, tt.args.epsGrad) {
				t.Errorf("ExternalPenalty() gotYmin = %v, want %v", gotYmin, tt.wantYmin)
			}
			if !equal(gotFmin, tt.wantFmin, tt.args.epsGrad) {
				t.Errorf("ExternalPenalty() gotFmin = %v, want %v", gotFmin, tt.wantFmin)
			}
			if !equal(gotR, tt.wantR, tt.args.epsGrad) {
				t.Errorf("ExternalPenalty() gotR = %v, want %v", gotR, tt.wantR)
			}
			if gotOuterIt != tt.wantOuterIt {
				t.Errorf("ExternalPenalty() gotOuterIt = %v, want %v", gotOuterIt, tt.wantOuterIt)
			}
		})
	}
}
