// Package voiceorder provides the viam:conversation-bundle:voice-order model —
// an orchestrator that listens for a wake word on a microphone, transcribes
// the following utterance, interprets it with Claude, dispatches a structured
// order to a target service via DoCommand, and speaks the reply through a
// text-to-speech service.
package voiceorder

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	speech "cloud.google.com/go/speech/apiv1"
	speechpb "cloud.google.com/go/speech/apiv1/speechpb"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"

	"go.viam.com/rdk/components/audioin"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"

	googleoption "google.golang.org/api/option"
)

var Model = resource.NewModel("viam", "conversation-bundle", "voice-order")

func init() {
	resource.RegisterService(generic.API, Model,
		resource.Registration[resource.Resource, *Config]{
			Constructor: newService,
		},
	)
}

type Config struct {
	AudioInName       string `json:"audio_in"`
	TextToSpeechName  string `json:"text_to_speech"`
	TargetServiceName string `json:"target_service"`
	TargetCommandKey  string `json:"target_command"`

	WakeWord        string  `json:"wake_word,omitempty"`
	WakeCooldownSec float64 `json:"wake_cooldown_sec,omitempty"`

	GoogleCredJSON map[string]interface{} `json:"google_credentials_json"`
	LanguageCode   string                 `json:"language_code,omitempty"`

	AnthropicAPIKey string `json:"anthropic_api_key"`
	LLMModel        string `json:"llm_model,omitempty"`
	SystemPrompt    string `json:"system_prompt"`
	MaxTokens       int    `json:"max_tokens,omitempty"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.AudioInName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "audio_in")
	}
	if cfg.TextToSpeechName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "text_to_speech")
	}
	if cfg.TargetServiceName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "target_service")
	}
	if cfg.TargetCommandKey == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "target_command")
	}
	if len(cfg.GoogleCredJSON) == 0 {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "google_credentials_json")
	}
	if cfg.AnthropicAPIKey == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "anthropic_api_key")
	}
	if cfg.SystemPrompt == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "system_prompt")
	}
	return []string{cfg.AudioInName, cfg.TextToSpeechName, cfg.TargetServiceName}, nil, nil
}

type service struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	mic           audioin.AudioIn
	tts           resource.Resource
	target        resource.Resource
	targetCmdKey  string

	wakeWord      string
	wakeCooldown  time.Duration
	languageCode  string
	systemPrompt  string
	maxTokens     int64
	anthropicModel anthropic.Model

	speechClient    *speech.Client
	anthropicClient anthropic.Client

	workerCtx    context.Context
	workerCancel context.CancelFunc
	workerWG     sync.WaitGroup

	mu       sync.Mutex
	lastWake time.Time
}

func newService(ctx context.Context, deps resource.Dependencies, rawConf resource.Config, logger logging.Logger) (resource.Resource, error) {
	conf, err := resource.NativeConfig[*Config](rawConf)
	if err != nil {
		return nil, err
	}
	return New(ctx, deps, rawConf.ResourceName(), conf, logger)
}

func New(ctx context.Context, deps resource.Dependencies, name resource.Name, conf *Config, logger logging.Logger) (resource.Resource, error) {
	mic, err := audioin.FromProvider(deps, conf.AudioInName)
	if err != nil {
		return nil, fmt.Errorf("audio_in %q not found: %w", conf.AudioInName, err)
	}
	tts, err := deps.Lookup(generic.Named(conf.TextToSpeechName))
	if err != nil {
		return nil, fmt.Errorf("text_to_speech %q not found: %w", conf.TextToSpeechName, err)
	}
	target, err := deps.Lookup(generic.Named(conf.TargetServiceName))
	if err != nil {
		return nil, fmt.Errorf("target_service %q not found: %w", conf.TargetServiceName, err)
	}

	credBytes, err := json.Marshal(conf.GoogleCredJSON)
	if err != nil {
		return nil, fmt.Errorf("marshal google credentials: %w", err)
	}
	speechClient, err := speech.NewClient(ctx, googleoption.WithCredentialsJSON(credBytes))
	if err != nil {
		return nil, fmt.Errorf("google speech client: %w", err)
	}

	anthropicClient := anthropic.NewClient(anthropicoption.WithAPIKey(conf.AnthropicAPIKey))

	wake := conf.WakeWord
	if wake == "" {
		wake = "hey beanjamin"
	}
	cooldown := time.Duration(conf.WakeCooldownSec * float64(time.Second))
	if cooldown <= 0 {
		cooldown = 2 * time.Second
	}
	lang := conf.LanguageCode
	if lang == "" {
		lang = "en-US"
	}
	maxTokens := int64(conf.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 512
	}
	model := anthropic.Model(conf.LLMModel)
	if model == "" {
		model = anthropic.Model("claude-opus-4-7")
	}

	s := &service{
		name:            name,
		logger:          logger,
		mic:             mic,
		tts:             tts,
		target:          target,
		targetCmdKey:    conf.TargetCommandKey,
		wakeWord:        strings.ToLower(wake),
		wakeCooldown:    cooldown,
		languageCode:    lang,
		systemPrompt:    conf.SystemPrompt,
		maxTokens:       maxTokens,
		anthropicModel:  model,
		speechClient:    speechClient,
		anthropicClient: anthropicClient,
	}
	s.workerCtx, s.workerCancel = context.WithCancel(context.Background())
	s.workerWG.Add(1)
	go s.run()
	return s, nil
}

func (s *service) Name() resource.Name { return s.name }

func (s *service) Status(ctx context.Context) (map[string]interface{}, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return map[string]interface{}{
		"wake_word": s.wakeWord,
		"last_wake": s.lastWake.Format(time.RFC3339Nano),
		"llm_model": string(s.anthropicModel),
		"language":  s.languageCode,
	}, nil
}

func (s *service) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	if _, ok := cmd["status"]; ok {
		s.mu.Lock()
		defer s.mu.Unlock()
		return map[string]interface{}{
			"wake_word":   s.wakeWord,
			"last_wake":   s.lastWake.Format(time.RFC3339Nano),
			"llm_model":   string(s.anthropicModel),
			"language":    s.languageCode,
		}, nil
	}
	return nil, fmt.Errorf("unknown command; supported: status")
}

func (s *service) Close(ctx context.Context) error {
	if s.workerCancel != nil {
		s.workerCancel()
	}
	s.workerWG.Wait()
	if s.speechClient != nil {
		return s.speechClient.Close()
	}
	return nil
}

// run is the top-level listening loop. Each iteration opens a fresh STT
// streaming session, waits for a wake-word + utterance, processes the order,
// speaks the reply, and loops back.
func (s *service) run() {
	defer s.workerWG.Done()
	for s.workerCtx.Err() == nil {
		utterance, err := s.listenForOrder(s.workerCtx)
		if err != nil {
			if s.workerCtx.Err() != nil {
				return
			}
			// Google STT caps streaming sessions at 305s. The outer loop
			// reopens a fresh stream on the next iteration, so this is
			// routine, not an error.
			if strings.Contains(err.Error(), "Exceeded maximum allowed stream duration") {
				s.logger.Infow("STT session hit Google's 305s limit; reopening", "err", err)
			} else {
				s.logger.Errorw("listen loop error", "err", err)
			}
			select {
			case <-s.workerCtx.Done():
				return
			case <-time.After(time.Second):
			}
			continue
		}
		if utterance == "" {
			continue
		}
		s.logger.Infow("captured utterance", "text", utterance)
		reply, err := s.interpret(s.workerCtx, utterance)
		if err != nil {
			s.logger.Errorw("llm interpret failed", "err", err)
			s.speak(s.workerCtx, "Sorry, I had trouble understanding that. Could you try again?")
			continue
		}
		if reply.Response != "" {
			s.speak(s.workerCtx, reply.Response)
		}
		if reply.Order != nil {
			if err := s.dispatchOrder(s.workerCtx, reply.Order); err != nil {
				s.logger.Errorw("dispatch order failed", "err", err, "order", reply.Order)
			}
		}
	}
}

// listenForOrder opens a Google STT streaming session, pipes mic audio to it,
// waits for the wake word in a live transcript, then returns the remainder of
// that utterance once Google marks it final.
func (s *service) listenForOrder(ctx context.Context) (string, error) {
	chunks, err := s.mic.GetAudio(ctx, "pcm16", 0, 0, nil)
	if err != nil {
		return "", fmt.Errorf("open mic: %w", err)
	}

	// Read the first chunk out-of-band to learn the sample rate.
	var first *audioin.AudioChunk
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case c, ok := <-chunks:
		if !ok {
			return "", fmt.Errorf("mic stream closed before any audio")
		}
		first = c
	}
	if first.AudioInfo == nil {
		return "", fmt.Errorf("mic did not report AudioInfo; cannot configure STT")
	}
	sampleRate := first.AudioInfo.SampleRateHz

	sttStream, err := s.speechClient.StreamingRecognize(ctx)
	if err != nil {
		return "", fmt.Errorf("open STT stream: %w", err)
	}
	// Send config first.
	if err := sttStream.Send(&speechpb.StreamingRecognizeRequest{
		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
			StreamingConfig: &speechpb.StreamingRecognitionConfig{
				Config: &speechpb.RecognitionConfig{
					Encoding:        speechpb.RecognitionConfig_LINEAR16,
					SampleRateHertz: sampleRate,
					LanguageCode:    s.languageCode,
				},
				InterimResults:  true,
				SingleUtterance: false,
			},
		},
	}); err != nil {
		return "", fmt.Errorf("send STT config: %w", err)
	}

	// Piper goroutine: drain mic chunks into the STT stream until ctx done,
	// mic closes, or the outer function signals done via pipeDone.
	pipeCtx, pipeCancel := context.WithCancel(ctx)
	defer pipeCancel()
	var pipeWG sync.WaitGroup
	pipeWG.Add(1)
	go func() {
		defer pipeWG.Done()
		// Forward the first chunk we already read.
		if err := sttStream.Send(&speechpb.StreamingRecognizeRequest{
			StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{AudioContent: first.AudioData},
		}); err != nil {
			return
		}
		for {
			select {
			case <-pipeCtx.Done():
				return
			case c, ok := <-chunks:
				if !ok {
					return
				}
				if err := sttStream.Send(&speechpb.StreamingRecognizeRequest{
					StreamingRequest: &speechpb.StreamingRecognizeRequest_AudioContent{AudioContent: c.AudioData},
				}); err != nil {
					return
				}
			}
		}
	}()

	// State machine on STT responses.
	state := stateIdle
	var orderText strings.Builder

	defer func() {
		pipeCancel()
		_ = sttStream.CloseSend()
		pipeWG.Wait()
	}()

	for {
		resp, err := sttStream.Recv()
		if err == io.EOF {
			return "", nil
		}
		if err != nil {
			return "", fmt.Errorf("STT recv: %w", err)
		}
		for _, result := range resp.Results {
			if len(result.Alternatives) == 0 {
				continue
			}
			transcript := result.Alternatives[0].Transcript
			lower := strings.ToLower(transcript)

			s.logger.Debugw("stt transcript",
				"text", transcript,
				"is_final", result.IsFinal,
				"state", stateName(state))

			switch state {
			case stateIdle:
				idx := strings.Index(lower, s.wakeWord)
				if idx < 0 {
					continue
				}
				s.mu.Lock()
				if time.Since(s.lastWake) < s.wakeCooldown {
					s.mu.Unlock()
					continue
				}
				s.lastWake = time.Now()
				s.mu.Unlock()
				s.logger.Infow("wake word detected", "transcript", transcript)
				state = stateListening
				// If this is already a final result carrying the post-wake
				// order ("hey beanjamin, espresso please"), short-circuit and
				// return it now.
				after := strings.TrimSpace(transcript[idx+len(s.wakeWord):])
				if result.IsFinal && after != "" {
					return after, nil
				}
				if after != "" {
					orderText.Reset()
					orderText.WriteString(after)
				}
			case stateListening:
				if result.IsFinal {
					// Strip the wake word (and anything before it) from the
					// final transcript. In a noisy room Google STT glues
					// ambient chatter into the same utterance window, so
					// "...orientation vector hey benjamin can I have an
					// espresso" needs to become just "can I have an espresso"
					// before we hand it to the LLM.
					trimmed := strings.TrimSpace(transcript)
					if idx := strings.LastIndex(strings.ToLower(trimmed), s.wakeWord); idx >= 0 {
						after := strings.TrimSpace(trimmed[idx+len(s.wakeWord):])
						if after != "" {
							return after, nil
						}
					}
					if trimmed != "" {
						return trimmed, nil
					}
					return strings.TrimSpace(orderText.String()), nil
				}
			}
		}
	}
}

type state int

const (
	stateIdle state = iota
	stateListening
)

func stateName(s state) string {
	switch s {
	case stateIdle:
		return "idle"
	case stateListening:
		return "listening"
	}
	return "unknown"
}

// interpretedReply is what the LLM is asked to return.
type interpretedReply struct {
	Order    map[string]interface{} `json:"order"`
	Response string                 `json:"response"`
}

func (s *service) interpret(ctx context.Context, utterance string) (interpretedReply, error) {
	msg, err := s.anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     s.anthropicModel,
		MaxTokens: s.maxTokens,
		System: []anthropic.TextBlockParam{{
			Text:         s.systemPrompt,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(utterance)),
		},
	})
	if err != nil {
		return interpretedReply{}, fmt.Errorf("anthropic messages: %w", err)
	}

	var text strings.Builder
	for _, block := range msg.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(tb.Text)
		}
	}
	raw := strings.TrimSpace(text.String())
	s.logger.Debugw("llm raw response",
		"text", raw,
		"cache_read", msg.Usage.CacheReadInputTokens,
		"cache_write", msg.Usage.CacheCreationInputTokens)

	// Accept the response either as a bare JSON object or as JSON wrapped in
	// a ```json code fence — Claude sometimes adds one despite instructions.
	body := stripCodeFence(raw)
	var reply interpretedReply
	if err := json.Unmarshal([]byte(body), &reply); err != nil {
		// Fallback: treat the whole text as the spoken response and emit no
		// order. Better than a failed turn.
		return interpretedReply{Response: raw}, nil
	}
	return reply, nil
}

func (s *service) speak(ctx context.Context, text string) {
	if text == "" || s.tts == nil {
		return
	}
	if _, err := s.tts.DoCommand(ctx, map[string]interface{}{"say": text}); err != nil {
		s.logger.Errorw("tts say failed", "err", err, "text", text)
	}
}

func (s *service) dispatchOrder(ctx context.Context, order map[string]interface{}) error {
	_, err := s.target.DoCommand(ctx, map[string]interface{}{s.targetCmdKey: order})
	return err
}

// stripCodeFence removes a surrounding ```…``` fence (with or without a
// language tag) if present.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// Drop the opening fence line.
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		s = s[nl+1:]
	}
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	return strings.TrimSpace(s)
}
