// Package texttospeech provides the viam:conversation-bundle:text-to-speech model.
package texttospeech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"

	texttospeech "cloud.google.com/go/texttospeech/apiv1"
	texttospeechpb "cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
	"go.viam.com/rdk/components/audioout"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
	"go.viam.com/rdk/utils"
	"google.golang.org/api/option"
)

// Google Cloud TTS returns LINEAR16 audio at 24 kHz mono by default.
const defaultSampleRateHz = 24000

// asyncQueueSize caps how many pending say_async requests can be buffered.
const asyncQueueSize = 64

var Model = resource.NewModel("viam", "conversation-bundle", "text-to-speech")

func init() {
	resource.RegisterService(generic.API, Model,
		resource.Registration[resource.Resource, *Config]{
			Constructor: newService,
		},
	)
}

type Config struct {
	AudioOutName   string                 `json:"audio_out"`
	LanguageCode   string                 `json:"language_code,omitempty"`
	VoiceName      string                 `json:"voice_name,omitempty"`
	GoogleCredJSON map[string]interface{} `json:"google_credentials_json"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.AudioOutName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "audio_out")
	}
	if len(cfg.GoogleCredJSON) == 0 {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "google_credentials_json")
	}
	return []string{cfg.AudioOutName}, nil, nil
}

type ttsService struct {
	resource.AlwaysRebuild

	name         resource.Name
	logger       logging.Logger
	audioOut     audioout.AudioOut
	ttsClient    *texttospeech.Client
	languageCode string
	voiceName    string

	// playMu serializes audio playback so async speech waits whenever any
	// other speech (sync or async) is currently being played.
	playMu sync.Mutex

	// asyncQueue buffers pending say_async requests for the background worker.
	asyncQueue chan string
	// workerCtx/workerCancel control the lifetime of the async worker
	// goroutine and any in-flight synthesis/playback it is running.
	workerCtx    context.Context
	workerCancel context.CancelFunc
	workerWG     sync.WaitGroup
}

func newService(ctx context.Context, deps resource.Dependencies, rawConf resource.Config, logger logging.Logger) (resource.Resource, error) {
	conf, err := resource.NativeConfig[*Config](rawConf)
	if err != nil {
		return nil, err
	}
	return New(ctx, deps, rawConf.ResourceName(), conf, logger)
}

func New(ctx context.Context, deps resource.Dependencies, name resource.Name, conf *Config, logger logging.Logger) (resource.Resource, error) {
	ao, err := audioout.FromProvider(deps, conf.AudioOutName)
	if err != nil {
		return nil, fmt.Errorf("audio_out %q not found in dependencies: %w", conf.AudioOutName, err)
	}

	credBytes, err := json.Marshal(conf.GoogleCredJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Google credentials: %w", err)
	}

	ttsClient, err := texttospeech.NewClient(ctx,
		option.WithAuthCredentialsJSON(option.ServiceAccount, credBytes),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Google TTS client: %w", err)
	}

	lang := conf.LanguageCode
	if lang == "" {
		lang = "en-US"
	}

	s := &ttsService{
		name:         name,
		logger:       logger,
		audioOut:     ao,
		ttsClient:    ttsClient,
		languageCode: lang,
		voiceName:    conf.VoiceName,
		asyncQueue:   make(chan string, asyncQueueSize),
	}
	// The worker must outlive the constructor's ctx, so derive from Background
	// and tear down explicitly in Close.
	s.workerCtx, s.workerCancel = context.WithCancel(context.Background())
	s.workerWG.Add(1)
	go s.asyncWorker()

	return s, nil
}

func (s *ttsService) Name() resource.Name {
	return s.name
}

func (s *ttsService) Say(ctx context.Context, text string) (string, error) {
	if err := s.synthesizeAndPlay(ctx, text); err != nil {
		return "", err
	}
	return text, nil
}

// synthesizeAndPlay synthesizes text via Google TTS and plays the resulting
// audio through the speaker, serialized behind playMu so no two playbacks
// overlap. Synthesis happens outside the mutex so queued requests can be
// prepared while another playback is still in progress.
func (s *ttsService) synthesizeAndPlay(ctx context.Context, text string) error {
	s.logger.Infof("synthesising: %q", text)

	voice := &texttospeechpb.VoiceSelectionParams{
		LanguageCode: s.languageCode,
	}
	// If voice name is blank, Google picks a default voice for the language.
	if s.voiceName != "" {
		voice.Name = s.voiceName
	}

	resp, err := s.ttsClient.SynthesizeSpeech(ctx, &texttospeechpb.SynthesizeSpeechRequest{
		Input:       &texttospeechpb.SynthesisInput{InputSource: &texttospeechpb.SynthesisInput_Text{Text: text}},
		Voice:       voice,
		AudioConfig: &texttospeechpb.AudioConfig{AudioEncoding: texttospeechpb.AudioEncoding_LINEAR16},
	})
	if err != nil {
		return fmt.Errorf("Google TTS synthesis failed: %w", err)
	}

	pcm := stripWAVHeader(resp.AudioContent)
	stereo := monoToStereo(pcm)

	s.playMu.Lock()
	defer s.playMu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}

	if err := s.audioOut.Play(ctx, stereo, &utils.AudioInfo{
		Codec:        utils.CodecPCM16,
		SampleRateHz: defaultSampleRateHz,
		NumChannels:  2,
	}, nil); err != nil {
		return fmt.Errorf("audio_out play failed: %w", err)
	}
	return nil
}

// asyncWorker drains the async queue one item at a time, synthesizing and
// playing each text sequentially. Because it pulls a single item at a time
// and playback is serialized behind playMu, a queued say_async will only
// reach the speaker once any prior speech (sync or async) has finished.
func (s *ttsService) asyncWorker() {
	defer s.workerWG.Done()
	for {
		select {
		case <-s.workerCtx.Done():
			return
		case text := <-s.asyncQueue:
			if err := s.synthesizeAndPlay(s.workerCtx, text); err != nil {
				if s.workerCtx.Err() != nil {
					return
				}
				s.logger.Errorf("async say failed for %q: %v", text, err)
			}
		}
	}
}

func (s *ttsService) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	if text, ok := cmd["say"].(string); ok {
		result, err := s.Say(ctx, text)
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"text": result}, nil
	}
	if text, ok := cmd["say_async"].(string); ok {
		select {
		case s.asyncQueue <- text:
			return map[string]interface{}{"queued": text}, nil
		default:
			return nil, fmt.Errorf("async speech queue is full (capacity %d)", asyncQueueSize)
		}
	}
	return nil, fmt.Errorf("unknown command, supported commands: say, say_async")
}

// stripWAVHeader removes a WAV/RIFF header if present, returning raw PCM data.
func stripWAVHeader(data []byte) []byte {
	if len(data) > 44 && string(data[:4]) == "RIFF" {
		return data[44:]
	}
	return data
}

// monoToStereo duplicates each LINEAR16 sample so mono PCM becomes stereo.
func monoToStereo(mono []byte) []byte {
	stereo := make([]byte, len(mono)*2)
	for i := 0; i < len(mono)-1; i += 2 {
		sample := binary.LittleEndian.Uint16(mono[i:])
		binary.LittleEndian.PutUint16(stereo[i*2:], sample)
		binary.LittleEndian.PutUint16(stereo[i*2+2:], sample)
	}
	return stereo
}

func (s *ttsService) Status(ctx context.Context) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

func (s *ttsService) Close(ctx context.Context) error {
	if s.workerCancel != nil {
		s.workerCancel()
	}
	s.workerWG.Wait()
	if s.ttsClient != nil {
		return s.ttsClient.Close()
	}
	return nil
}
