package voicecommand

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
)

// fakeStatusSource is a command_status resource whose Status() is supplied by
// the test and whose calls are counted.
type fakeStatusSource struct {
	calls  atomic.Int32
	status func(ctx context.Context) (map[string]interface{}, error)
}

func (f *fakeStatusSource) Name() resource.Name { return generic.Named("status-source") }

func (f *fakeStatusSource) DoCommand(context.Context, map[string]interface{}) (map[string]interface{}, error) {
	return nil, nil
}

func (f *fakeStatusSource) Status(ctx context.Context) (map[string]interface{}, error) {
	f.calls.Add(1)
	return f.status(ctx)
}

func (f *fakeStatusSource) Close(context.Context) error { return nil }

func newStatusTestService(t *testing.T, src *fakeStatusSource) *service {
	t.Helper()
	s := &service{
		logger:        logging.NewTestLogger(t),
		commandStatus: &commandStatusRef{name: "status-source", handle: src},
	}
	s.workerCtx, s.workerCancel = context.WithCancel(context.Background())
	t.Cleanup(s.workerCancel)
	return s
}

func TestFetchContextBoundsAHangingStatus(t *testing.T) {
	src := &fakeStatusSource{status: func(ctx context.Context) (map[string]interface{}, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	s := newStatusTestService(t, src)

	start := time.Now()
	_, status := s.fetchContext(context.Background())
	elapsed := time.Since(start)

	if elapsed > contextFetchTimeout+time.Second {
		t.Fatalf("fetchContext took %v, want about %v", elapsed, contextFetchTimeout)
	}
	if _, ok := status["error"]; !ok {
		t.Fatalf("status = %v, want an error slot", status)
	}
}

func TestFetchCommandFaultReportsWhetherTheFlagIsPublished(t *testing.T) {
	cases := []struct {
		name         string
		status       map[string]interface{}
		err          error
		wantActive   bool
		wantReported bool
	}{
		{"raised", map[string]interface{}{"fault_active": true}, nil, true, true},
		{"lowered", map[string]interface{}{"fault_active": false}, nil, false, true},
		{"not published", map[string]interface{}{"recognized": false}, nil, false, false},
		{"failed", nil, errors.New("boom"), false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := &fakeStatusSource{status: func(context.Context) (map[string]interface{}, error) {
				return tc.status, tc.err
			}}
			s := newStatusTestService(t, src)
			active, reported, err := s.fetchCommandFault()
			if active != tc.wantActive || reported != tc.wantReported || (err != nil) != (tc.err != nil) {
				t.Fatalf("fetchCommandFault = %v, %v, %v; want %v, %v, err %v",
					active, reported, err, tc.wantActive, tc.wantReported, tc.err)
			}
		})
	}
}

// pollFor runs pollCommandFault for d and returns how many Status() calls it
// made.
func pollFor(t *testing.T, src *fakeStatusSource, d time.Duration) int32 {
	t.Helper()
	s := newStatusTestService(t, src)
	s.workerWG.Add(1)
	go s.pollCommandFault()
	time.Sleep(d)
	s.workerCancel()
	s.workerWG.Wait()
	return src.calls.Load()
}

func TestPollCommandFaultBacksOffFromSourcesWithoutTheFlag(t *testing.T) {
	window := 3 * commandFaultPollInterval

	publishing := &fakeStatusSource{status: func(context.Context) (map[string]interface{}, error) {
		return map[string]interface{}{"fault_active": false}, nil
	}}
	if got := pollFor(t, publishing, window); got < 3 {
		t.Errorf("source publishing fault_active polled %d times in %v, want at least 3", got, window)
	}

	silent := &fakeStatusSource{status: func(context.Context) (map[string]interface{}, error) {
		return map[string]interface{}{"recognized": false}, nil
	}}
	if got := pollFor(t, silent, window); got != 1 {
		t.Errorf("source without fault_active polled %d times in %v, want 1", got, window)
	}

	failing := &fakeStatusSource{status: func(context.Context) (map[string]interface{}, error) {
		return nil, errors.New("deadline exceeded")
	}}
	if got := pollFor(t, failing, window); got != 1 {
		t.Errorf("failing source polled %d times in %v, want 1", got, window)
	}
}
