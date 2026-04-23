// Package voicecommand provides the viam:conversation-bundle:voice-command
// model — an orchestrator that listens for a wake word on a microphone,
// transcribes the following utterance, classifies it against a configured
// list of commands via Claude, dispatches the matched command's DoCommand
// payload to its target resource, and speaks the reply through a
// text-to-speech service.
package voicecommand

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
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

var Model = resource.NewModel("viam", "conversation-bundle", "voice-command")

func init() {
	resource.RegisterService(generic.API, Model,
		resource.Registration[resource.Resource, *Config]{
			Constructor: newService,
		},
	)
}

// CommandEntry is one user-facing action voice-command can dispatch. The
// LLM sees Name + Description and returns the Name of the entry it chose
// (or nothing, to take no action); voice-command looks the entry up and
// dispatches DoCommand verbatim to the named Resource.
type CommandEntry struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Resource    string                 `json:"resource"`
	DoCommand   map[string]interface{} `json:"do_command"`
}

type Config struct {
	AudioInName      string `json:"audio_in"`
	TextToSpeechName string `json:"text_to_speech"`

	Commands []CommandEntry `json:"commands"`

	WakeWord        string  `json:"wake_word,omitempty"`
	WakeCooldownSec float64 `json:"wake_cooldown_sec,omitempty"`

	// ListenTimeoutSec caps how long we wait for the user's utterance
	// once we've begun actively listening — either right after wake-word
	// detection (wake mode) or throughout a conversation follow-up
	// (conversation mode). If the user stays silent for this long, we
	// stop listening and return to idle. Default 7 seconds.
	ListenTimeoutSec float64 `json:"listen_timeout_sec,omitempty"`

	// MaxUtteranceWords caps how many words we accumulate (post-wake-word
	// in wake mode, whole transcript in conversation mode) before we stop
	// waiting for Google STT's is_final and commit the partial transcript
	// to the LLM. Keeps latency bounded in noisy rooms where ambient
	// chatter keeps the utterance window open indefinitely. Default 10.
	MaxUtteranceWords int `json:"max_utterance_words,omitempty"`

	// ConversationEndCue is spoken when an active conversation window ends
	// (either because Claude signaled end or the timer expired) to remind
	// the user they'll need to say the wake word to re-engage. Omit the
	// field to use the default (a reminder naming the wake word); set it
	// to an empty string to suppress the cue entirely.
	ConversationEndCue *string `json:"conversation_end_cue,omitempty"`

	GoogleCredJSON map[string]interface{} `json:"google_credentials_json"`
	LanguageCode   string                 `json:"language_code,omitempty"`

	AnthropicAPIKey string `json:"anthropic_api_key"`
	LLMModel        string `json:"llm_model,omitempty"`
	// SystemPrompt is an optional free-form suffix appended to the
	// auto-generated system prompt (which already contains the command list
	// and output-format instructions). Use it for tone/style guidance
	// ("respond warmly", "keep replies under 20 words", etc.).
	SystemPrompt string `json:"system_prompt,omitempty"`
	MaxTokens    int    `json:"max_tokens,omitempty"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.AudioInName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "audio_in")
	}
	if cfg.TextToSpeechName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "text_to_speech")
	}
	if len(cfg.GoogleCredJSON) == 0 {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "google_credentials_json")
	}
	if cfg.AnthropicAPIKey == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "anthropic_api_key")
	}
	if len(cfg.Commands) == 0 {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "commands")
	}

	seenNames := map[string]bool{}
	resourceSet := map[string]bool{}
	for i, c := range cfg.Commands {
		if c.Name == "" {
			return nil, nil, fmt.Errorf("%s: commands[%d].name is required", path, i)
		}
		if seenNames[c.Name] {
			return nil, nil, fmt.Errorf("%s: commands[%d].name %q is not unique", path, i, c.Name)
		}
		seenNames[c.Name] = true
		if c.Description == "" {
			return nil, nil, fmt.Errorf("%s: commands[%d].description is required", path, i)
		}
		if c.Resource == "" {
			return nil, nil, fmt.Errorf("%s: commands[%d].resource is required", path, i)
		}
		if len(c.DoCommand) == 0 {
			return nil, nil, fmt.Errorf("%s: commands[%d].do_command is required", path, i)
		}
		resourceSet[c.Resource] = true
	}

	deps := []string{cfg.AudioInName, cfg.TextToSpeechName}
	resources := make([]string, 0, len(resourceSet))
	for r := range resourceSet {
		resources = append(resources, r)
	}
	sort.Strings(resources)
	deps = append(deps, resources...)
	return deps, nil, nil
}

type service struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	mic audioin.AudioIn
	tts resource.Resource

	// commands is keyed by CommandEntry.Name. resources is keyed by the
	// Resource name referenced from command entries.
	commands  map[string]CommandEntry
	resources map[string]resource.Resource

	wakeWord       string
	wakeCooldown   time.Duration
	listenTimeout   time.Duration
	maxWords       int
	endCue         string
	languageCode   string
	systemPrompt   string
	maxTokens      int64
	anthropicModel anthropic.Model

	speechClient    *speech.Client
	anthropicClient anthropic.Client

	workerCtx    context.Context
	workerCancel context.CancelFunc
	workerWG     sync.WaitGroup

	mu       sync.Mutex
	lastWake time.Time
	// convoExpiresAt is non-zero while a conversation window is open.
	// During the window, voice-command skips wake-word gating and passes
	// convoHistory to the LLM so follow-up utterances carry context.
	convoExpiresAt time.Time
	convoHistory   []anthropic.MessageParam
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

	commands := make(map[string]CommandEntry, len(conf.Commands))
	resources := map[string]resource.Resource{}
	for _, c := range conf.Commands {
		commands[c.Name] = c
		if _, ok := resources[c.Resource]; ok {
			continue
		}
		res, err := deps.Lookup(generic.Named(c.Resource))
		if err != nil {
			return nil, fmt.Errorf("command %q: resource %q not found: %w", c.Name, c.Resource, err)
		}
		resources[c.Resource] = res
	}

	credBytes, err := json.Marshal(conf.GoogleCredJSON)
	if err != nil {
		return nil, fmt.Errorf("marshal google credentials: %w", err)
	}
	speechClient, err := speech.NewClient(ctx,
		googleoption.WithAuthCredentialsJSON(googleoption.ServiceAccount, credBytes),
	)
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
	listenTimeout := time.Duration(conf.ListenTimeoutSec * float64(time.Second))
	if listenTimeout <= 0 {
		listenTimeout = 7 * time.Second
	}
	maxWords := conf.MaxUtteranceWords
	if maxWords <= 0 {
		maxWords = 10
	}
	// Omitted = default cue naming the wake word; explicit "" = suppress.
	var endCue string
	if conf.ConversationEndCue == nil {
		endCue = fmt.Sprintf("Say %s when you need me.", wake)
	} else {
		endCue = *conf.ConversationEndCue
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

	systemPrompt := buildSystemPrompt(conf.Commands, conf.SystemPrompt)
	logger.Debugw("built system prompt", "prompt", systemPrompt)

	s := &service{
		name:            name,
		logger:          logger,
		mic:             mic,
		tts:             tts,
		commands:        commands,
		resources:       resources,
		wakeWord:        strings.ToLower(wake),
		wakeCooldown:    cooldown,
		listenTimeout:    listenTimeout,
		maxWords:        maxWords,
		endCue:          endCue,
		languageCode:    lang,
		systemPrompt:    systemPrompt,
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

// buildSystemPrompt assembles the system prompt Claude sees on every call.
// The auto-generated header stays in sync with the configured command list,
// so drift between config and prompt is impossible. The user-supplied
// suffix (Config.SystemPrompt) carries tone/style, not domain knowledge.
func buildSystemPrompt(cmds []CommandEntry, userSuffix string) string {
	var b strings.Builder
	b.WriteString("You are a voice-controlled assistant. A user just spoke to you. Pick at most one of the following commands to dispatch, and produce a short spoken reply.\n\n")
	b.WriteString("Available commands:\n")
	for _, c := range cmds {
		fmt.Fprintf(&b, "- %q — %s\n", c.Name, c.Description)
	}
	b.WriteString("\nRespond with exactly one JSON object and nothing else. No prose before or after, no markdown code fences:\n\n")
	b.WriteString("{\n")
	b.WriteString("  \"response\": \"<a short friendly sentence to say to the user>\",\n")
	b.WriteString("  \"command\": \"<name from the list, or null>\",\n")
	b.WriteString("  \"continue_conversation\": <true or false>\n")
	b.WriteString("}\n\n")
	b.WriteString("Rules:\n")
	b.WriteString("- Always produce a non-empty \"response\". Every turn gets a spoken reply — confirmations, acknowledgments, thank-yous, farewells, or even a good-natured \"sorry, didn't catch that\" for vague input. Never return an empty string.\n")
	b.WriteString("- If the utterance doesn't clearly match any command, set \"command\" to null and still say something natural in \"response\".\n")
	b.WriteString("- Set \"continue_conversation\" to true when you expect a follow-up — you asked a clarifying question or the exchange invites more back-and-forth. Set false when the exchange is clearly complete (e.g., the requested action was confirmed or the user said goodbye). Err on the side of true if unsure, so the user can chime back in.\n")
	b.WriteString("- If the utterance sounds like ambient conversation you weren't the intended recipient of, you can acknowledge it briefly (\"Sorry, were you talking to me?\" or similar) rather than going silent.\n")
	b.WriteString("- Prior turns in this conversation (if any) are included as chat history; treat them as context.")
	if strings.TrimSpace(userSuffix) != "" {
		b.WriteString("\n\n")
		b.WriteString(strings.TrimSpace(userSuffix))
	}
	return b.String()
}

func (s *service) Name() resource.Name { return s.name }

func (s *service) Status(ctx context.Context) (map[string]interface{}, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	names := make([]string, 0, len(s.commands))
	for n := range s.commands {
		names = append(names, n)
	}
	sort.Strings(names)
	return map[string]interface{}{
		"wake_word": s.wakeWord,
		"last_wake": s.lastWake.Format(time.RFC3339Nano),
		"llm_model": string(s.anthropicModel),
		"language":  s.languageCode,
		"commands":  names,
	}, nil
}

func (s *service) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	if _, ok := cmd["status"]; ok {
		return s.Status(ctx)
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

// inConversation reports whether a conversation window is currently open.
// Returns the history to thread into the next LLM call and the deadline by
// which the next utterance must arrive. When not in a conversation, history
// is nil and deadline is the zero time.
func (s *service) inConversation() (history []anthropic.MessageParam, deadline time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.convoExpiresAt.IsZero() || time.Now().After(s.convoExpiresAt) {
		return nil, time.Time{}
	}
	// Return a copy so callers don't see concurrent appends.
	h := make([]anthropic.MessageParam, len(s.convoHistory))
	copy(h, s.convoHistory)
	return h, s.convoExpiresAt
}

func (s *service) extendConversation(history []anthropic.MessageParam) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.convoExpiresAt = time.Now().Add(s.listenTimeout)
	s.convoHistory = history
}

func (s *service) endConversation() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.convoExpiresAt = time.Time{}
	s.convoHistory = nil
}

// run is the top-level listening loop. Each iteration opens a fresh STT
// streaming session, waits for either a wake-word + utterance (idle mode)
// or any utterance (conversation mode), classifies it with the LLM,
// dispatches the matched command (if any), and speaks the reply.
func (s *service) run() {
	defer s.workerWG.Done()
	for s.workerCtx.Err() == nil {
		history, deadline := s.inConversation()
		inConvo := !deadline.IsZero()

		listenCtx := s.workerCtx
		var cancel context.CancelFunc
		if inConvo {
			listenCtx, cancel = context.WithDeadline(s.workerCtx, deadline)
		}
		utterance, err := s.listenForCommand(listenCtx, inConvo)
		if cancel != nil {
			cancel()
		}

		if err != nil {
			if s.workerCtx.Err() != nil {
				return
			}
			// In conversation mode, a deadline-exceeded just means the window
			// timed out without a follow-up. End the conversation cleanly.
			if inConvo && (errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "context deadline exceeded")) {
				s.logger.Infow("conversation window timed out; returning to wake-word mode")
				s.speakEndCue(s.workerCtx)
				s.endConversation()
				continue
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
		s.logger.Infow("captured utterance", "text", utterance, "in_conversation", inConvo)

		reply, newHistory, err := s.interpret(s.workerCtx, utterance, history)
		if err != nil {
			s.logger.Errorw("llm interpret failed", "err", err)
			s.speak(s.workerCtx, "Sorry, I had trouble understanding that. Could you try again?")
			s.endConversation()
			continue
		}
		s.logger.Infow("llm decision",
			"command", reply.Command,
			"continue_conversation", reply.ContinueConversation,
			"response", reply.Response)

		if reply.Response != "" {
			s.speak(s.workerCtx, reply.Response)
		}
		if reply.Command != "" {
			if err := s.dispatch(s.workerCtx, reply.Command); err != nil {
				s.logger.Errorw("dispatch failed", "err", err, "command", reply.Command)
			}
		}
		if reply.ContinueConversation {
			s.extendConversation(newHistory)
		} else {
			// Play the wake-word reminder on every end-of-turn that
			// doesn't continue the conversation — covers both the
			// "one-shot wake-triggered turn" and "conversation ending
			// after multiple turns" cases, so the user always knows the
			// robot is idle and waiting for the next wake word.
			s.speakEndCue(s.workerCtx)
			s.endConversation()
		}
	}
}

// speakEndCue plays the configured wake-word reminder, if any, to let the
// user know the conversation window has closed. No-op if the cue was set
// to an empty string in config.
func (s *service) speakEndCue(ctx context.Context) {
	if s.endCue == "" {
		return
	}
	s.speak(ctx, s.endCue)
}

// listenForCommand opens a Google STT streaming session, pipes mic audio
// to it, waits for the wake word (or any utterance if startInConversation
// is true), and returns the captured utterance once Google marks it final
// (or earlier if the word-count cap is hit).
func (s *service) listenForCommand(ctx context.Context, startInConversation bool) (string, error) {
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

	// Open the STT stream under a cancellable child context so we can
	// enforce the listen-timeout by closing the stream from under Recv.
	streamCtx, streamCancel := context.WithCancel(ctx)
	defer streamCancel()
	sttStream, err := s.speechClient.StreamingRecognize(streamCtx)
	if err != nil {
		return "", fmt.Errorf("open STT stream: %w", err)
	}
	// Send config first.
	// Only enable Google's single_utterance mode in conversation follow-up
	// mode. There, it gives snappy turn-taking (~200-500ms after the user
	// stops talking). In wake mode we keep it OFF — otherwise every
	// ambient utterance from colleagues in the room terminates our STT
	// stream, forcing a ~4s reopen cycle during which we can't hear the
	// wake word. With single_utterance off, the stream stays open until
	// we close it (or the 305s limit); wake-mode latency is then capped
	// by our own listen-timeout (after wake-word detection) and the
	// max-utterance-words cap.
	if err := sttStream.Send(&speechpb.StreamingRecognizeRequest{
		StreamingRequest: &speechpb.StreamingRecognizeRequest_StreamingConfig{
			StreamingConfig: &speechpb.StreamingRecognitionConfig{
				Config: &speechpb.RecognitionConfig{
					Encoding:        speechpb.RecognitionConfig_LINEAR16,
					SampleRateHertz: sampleRate,
					LanguageCode:    s.languageCode,
				},
				InterimResults:  true,
				SingleUtterance: startInConversation,
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

	// State machine on STT responses. In conversation mode we skip
	// wake-word gating entirely — the next final transcript is the
	// user's follow-up.
	state := stateIdle
	if startInConversation {
		state = stateListening
	}
	var captureBuf strings.Builder

	// Listen-timeout timer: once we're in stateListening, the user has
	// listenTimeout to finish speaking. If the timer fires, we cancel
	// streamCtx, which makes Recv return a cancellation error that we
	// translate into a clean return with whatever we've captured.
	var listenTimer *time.Timer
	armListenTimeout := func() {
		if listenTimer != nil {
			return
		}
		listenTimer = time.AfterFunc(s.listenTimeout, streamCancel)
	}
	if startInConversation {
		armListenTimeout()
	}

	defer func() {
		if listenTimer != nil {
			listenTimer.Stop()
		}
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
			// If this error is due to our own listen-timeout cancelling
			// streamCtx (and the outer ctx is still alive), treat it as a
			// normal timeout: return whatever we captured without an error.
			if streamCtx.Err() != nil && ctx.Err() == nil {
				s.logger.Infow("listen timeout fired",
					"state", stateName(state),
					"captured", captureBuf.String())
				return strings.TrimSpace(captureBuf.String()), nil
			}
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
				armListenTimeout()
				after := strings.TrimSpace(transcript[idx+len(s.wakeWord):])
				// Short-circuit if we've already got enough to work with:
				// Google marked it final, OR the post-wake portion already
				// carries at least maxWords. Both keep us from waiting on
				// ambient chatter to finish.
				if after != "" && (result.IsFinal || wordCount(after) >= s.maxWords) {
					s.logger.Infow("committing utterance (wake-word short-circuit)",
						"is_final", result.IsFinal, "word_count", wordCount(after))
					return after, nil
				}
				if after != "" {
					captureBuf.Reset()
					captureBuf.WriteString(after)
				}
			case stateListening:
				trimmed := strings.TrimSpace(transcript)

				// Compute the post-wake portion (in wake mode) or the whole
				// transcript (in conversation mode) — this is what we'd
				// ultimately hand to the LLM.
				var post string
				if startInConversation {
					post = trimmed
				} else if idx := strings.LastIndex(strings.ToLower(trimmed), s.wakeWord); idx >= 0 {
					post = strings.TrimSpace(trimmed[idx+len(s.wakeWord):])
				} else {
					post = trimmed
				}

				// Keep captureBuf in sync with the latest interim so the
				// listen-timeout path has something to return if the timer
				// fires while the user is still mid-sentence.
				if post != "" {
					captureBuf.Reset()
					captureBuf.WriteString(post)
				}

				// Cut early on word count: once we've heard at least
				// maxWords of content post-wake, commit without waiting
				// for Google's is_final (which may be blocked indefinitely
				// by ambient chatter in noisy rooms).
				if post != "" && wordCount(post) >= s.maxWords {
					s.logger.Infow("committing utterance (word-count cap)",
						"word_count", wordCount(post), "cap", s.maxWords)
					return post, nil
				}

				if result.IsFinal {
					if post != "" {
						return post, nil
					}
					return strings.TrimSpace(captureBuf.String()), nil
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

// interpretedReply is what the LLM is asked to return. Command is the Name
// of a configured CommandEntry, or empty / null for "no action."
type interpretedReply struct {
	Response             string `json:"response"`
	Command              string `json:"command"`
	ContinueConversation bool   `json:"continue_conversation"`
}

// interpret sends the utterance (with optional prior conversation history)
// to Claude and returns the parsed reply plus the history extended with
// this turn's user and assistant messages. Callers can thread the returned
// history back into the next interpret() call to maintain context across
// a conversation window.
func (s *service) interpret(ctx context.Context, utterance string, history []anthropic.MessageParam) (interpretedReply, []anthropic.MessageParam, error) {
	userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(utterance))
	msgs := append(append([]anthropic.MessageParam{}, history...), userMsg)

	msg, err := s.anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     s.anthropicModel,
		MaxTokens: s.maxTokens,
		System: []anthropic.TextBlockParam{{
			Text:         s.systemPrompt,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}},
		Messages: msgs,
	})
	if err != nil {
		return interpretedReply{}, history, fmt.Errorf("anthropic messages: %w", err)
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
		"cache_write", msg.Usage.CacheCreationInputTokens,
		"history_turns", len(history))

	newHistory := append(msgs, msg.ToParam())

	// Accept the response either as a bare JSON object or as JSON wrapped in
	// a ```json code fence — Claude sometimes adds one despite instructions.
	body := stripCodeFence(raw)
	var reply interpretedReply
	if err := json.Unmarshal([]byte(body), &reply); err != nil {
		// Fallback: treat the whole text as the spoken response and emit no
		// command. Better than a failed turn.
		return interpretedReply{Response: raw}, newHistory, nil
	}
	return reply, newHistory, nil
}

func (s *service) speak(ctx context.Context, text string) {
	if text == "" || s.tts == nil {
		return
	}
	if _, err := s.tts.DoCommand(ctx, map[string]interface{}{"say": text}); err != nil {
		s.logger.Errorw("tts say failed", "err", err, "text", text)
	}
}

// dispatch looks up the named CommandEntry and fires its DoCommand payload at
// the pre-resolved target resource. Unknown command names are a soft error:
// we log and return nil so a hallucinated name doesn't break the listening
// loop.
func (s *service) dispatch(ctx context.Context, commandName string) error {
	entry, ok := s.commands[commandName]
	if !ok {
		s.logger.Warnw("llm chose unknown command; skipping dispatch", "command", commandName)
		return nil
	}
	target, ok := s.resources[entry.Resource]
	if !ok {
		// Should be unreachable — Validate + New ensure every entry's
		// Resource is in the map. Defensive log.
		return fmt.Errorf("resource %q for command %q missing from resource map", entry.Resource, commandName)
	}
	s.logger.Infow("dispatching command",
		"command", commandName,
		"resource", entry.Resource,
		"payload", entry.DoCommand)
	_, err := target.DoCommand(ctx, entry.DoCommand)
	return err
}

// wordCount counts whitespace-separated tokens. Good enough for the
// commit-early heuristic; we're not doing linguistics.
func wordCount(s string) int {
	return len(strings.Fields(s))
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
