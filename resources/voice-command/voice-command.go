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
	"sync/atomic"
	"time"

	speech "cloud.google.com/go/speech/apiv1"
	speechpb "cloud.google.com/go/speech/apiv1/speechpb"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"

	"go.viam.com/rdk/components/audioin"
	"go.viam.com/rdk/components/sensor"
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

// SensorEntry declares a Viam sensor resource whose Readings() are
// fetched fresh on every LLM call and injected into the prompt as a
// second, uncached system block — so Claude can reason about current
// state (queue depth, environmental conditions, etc.). Description
// helps Claude interpret the reading keys.
type SensorEntry struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// sensorRef is the runtime counterpart to SensorEntry: the resolved
// sensor.Sensor handle bundled with the metadata needed to format its
// readings into the prompt.
type sensorRef struct {
	name        string
	description string
	handle      sensor.Sensor
}

// commandStatusRef is the runtime counterpart to CommandStatusConfig: a
// resolved generic resource handle whose Status() method is fetched fresh
// on every turn.
type commandStatusRef struct {
	name        string
	description string
	handle      resource.Resource
}

// CommandStatusConfig declares a single Viam resource whose Status() method
// describes the current state of whatever command voice-command dispatched.
// Voice-command does not interpret the status itself — it just injects it
// into Claude's context under a distinct heading so Claude can factor
// command progress into its continue_conversation decision.
//
// The resource must be resolvable as a generic service (API
// rdk:service:generic). Status() is defined on every resource.Resource in
// rdk v0.119.2+; the default embedded implementation returns an empty map,
// so the source resource should implement its own Status() to be useful
// here.
type CommandStatusConfig struct {
	Resource    string `json:"resource"`
	Description string `json:"description,omitempty"`
}

type Config struct {
	AudioInName      string `json:"audio_in"`
	TextToSpeechName string `json:"text_to_speech"`

	Commands []CommandEntry `json:"commands"`

	// Sensors lists Viam sensor resources whose Readings() are fetched
	// fresh on every LLM call and attached to the request as additional
	// context. Omit or leave empty for no sensor injection. Fetches run
	// in parallel and fail soft — if a sensor errors, its slot in the
	// context reports the error so Claude can react appropriately.
	Sensors []SensorEntry `json:"sensors,omitempty"`

	// CommandStatus is an optional designated resource whose Status() method
	// reports the state of the last-dispatched command (e.g., whether it's
	// still running, queue depth, result so far). The Status map is included
	// in Claude's context every turn under a separate heading so Claude can
	// factor command progress into its continue_conversation decision — for
	// example, keeping the conversation open while a brew is in progress.
	// Voice-command does not interpret the status itself.
	CommandStatus *CommandStatusConfig `json:"command_status,omitempty"`

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

	// StableEndpointMs is the client-side silence grace period: once the
	// captured post-wake (or conversation) content stops changing, we wait
	// this long before committing — much shorter than Google's ~1-2s
	// internal VAD. Keeps TTFR low on short utterances like "espresso
	// please" without closing the STT stream. Default 500ms. Set to a
	// negative value to disable and fall back to is_final / maxWords only.
	StableEndpointMs int `json:"stable_endpoint_ms,omitempty"`

	// ConversationEndCue is spoken when an active conversation window ends
	// (either because Claude signaled end or the timer expired) to remind
	// the user they'll need to say the wake word to re-engage. Omit the
	// field to use the default (a reminder naming the wake word); set it
	// to an empty string to suppress the cue entirely.
	ConversationEndCue *string `json:"conversation_end_cue,omitempty"`

	// MinLullPrompts is the floor on how many Claude-generated nudge
	// prompts voice-command will produce on consecutive silences before
	// honoring Claude's continue_conversation=false signal and closing
	// the conversation. If Claude sets continue_conversation=false on a
	// nudge before this floor is reached, voice-command overrides and
	// keeps the conversation open. Defaults to 2.
	MinLullPrompts int `json:"min_lull_prompts,omitempty"`

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

	// Validate sensor entries and build their resource-name list.
	seenSensors := map[string]bool{}
	for i, se := range cfg.Sensors {
		if se.Name == "" {
			return nil, nil, fmt.Errorf("%s: sensors[%d].name is required", path, i)
		}
		if seenSensors[se.Name] {
			return nil, nil, fmt.Errorf("%s: sensors[%d].name %q is not unique", path, i, se.Name)
		}
		seenSensors[se.Name] = true
		if se.Description == "" {
			return nil, nil, fmt.Errorf("%s: sensors[%d].description is required", path, i)
		}
	}
	if cfg.CommandStatus != nil {
		if cfg.CommandStatus.Resource == "" {
			return nil, nil, fmt.Errorf("%s: command_status.resource is required", path)
		}
	}

	deps := []string{cfg.AudioInName, cfg.TextToSpeechName}
	if cfg.CommandStatus != nil {
		// Resolved under generic.API, same as commands[].resource — adding
		// as a bare name defaults to that API in the dep resolver.
		resourceSet[cfg.CommandStatus.Resource] = true
	}
	resources := make([]string, 0, len(resourceSet))
	for r := range resourceSet {
		resources = append(resources, r)
	}
	sort.Strings(resources)
	deps = append(deps, resources...)
	// Sensors are registered under the rdk:component:sensor API, so they
	// need the full-resource-name form (otherwise the Viam dep resolver
	// would look up a generic service by that name and fail).
	sensorNames := make([]string, 0, len(cfg.Sensors))
	for _, se := range cfg.Sensors {
		sensorNames = append(sensorNames, sensor.Named(se.Name).String())
	}
	sort.Strings(sensorNames)
	deps = append(deps, sensorNames...)
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

	// sensors retains config order so readings in the prompt list in a
	// stable order (helps Claude's caching of the sensor section).
	sensors []sensorRef

	// commandStatus is the optional designated "command status" resource —
	// its Status() output is injected into Claude's context under a distinct
	// heading so Claude can reason about command progress. nil if not
	// configured.
	commandStatus *commandStatusRef

	wakeWord       string
	wakeCooldown   time.Duration
	listenTimeout   time.Duration
	maxWords       int
	stableEndpoint time.Duration
	endCue         string
	minLullPrompts int
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
	// lullCount tracks consecutive silence-triggered nudge prompts within
	// the current conversation. Resets on a real user utterance and on
	// endConversation. Used to enforce MinLullPrompts.
	lullCount int
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

	// Resolve each configured sensor. Preserved in config order so their
	// readings always appear in a stable order in the prompt.
	sensors := make([]sensorRef, 0, len(conf.Sensors))
	for _, se := range conf.Sensors {
		handle, err := sensor.FromProvider(deps, se.Name)
		if err != nil {
			return nil, fmt.Errorf("sensor %q not found: %w", se.Name, err)
		}
		sensors = append(sensors, sensorRef{
			name:        se.Name,
			description: se.Description,
			handle:      handle,
		})
	}
	var commandStatus *commandStatusRef
	if conf.CommandStatus != nil {
		// Reuse the handle already resolved for commands[] if the same
		// resource is configured as both a command target and the status
		// source — common case, since the coffee service is typically both.
		handle, ok := resources[conf.CommandStatus.Resource]
		if !ok {
			h, err := deps.Lookup(generic.Named(conf.CommandStatus.Resource))
			if err != nil {
				return nil, fmt.Errorf("command_status resource %q not found: %w", conf.CommandStatus.Resource, err)
			}
			handle = h
		}
		commandStatus = &commandStatusRef{
			name:        conf.CommandStatus.Resource,
			description: conf.CommandStatus.Description,
			handle:      handle,
		}
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
	// StableEndpointMs == 0 means unset, use default. Negative disables.
	var stableEndpoint time.Duration
	switch {
	case conf.StableEndpointMs == 0:
		stableEndpoint = 500 * time.Millisecond
	case conf.StableEndpointMs < 0:
		stableEndpoint = 0
	default:
		stableEndpoint = time.Duration(conf.StableEndpointMs) * time.Millisecond
	}
	// Omitted = default cue naming the wake word; explicit "" = suppress.
	var endCue string
	if conf.ConversationEndCue == nil {
		endCue = fmt.Sprintf("Say %s when you need me.", wake)
	} else {
		endCue = *conf.ConversationEndCue
	}
	minLullPrompts := conf.MinLullPrompts
	if minLullPrompts <= 0 {
		minLullPrompts = 2
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
		sensors:             sensors,
		commandStatus:       commandStatus,
		wakeWord:        strings.ToLower(wake),
		wakeCooldown:    cooldown,
		listenTimeout:    listenTimeout,
		maxWords:        maxWords,
		stableEndpoint:  stableEndpoint,
		endCue:          endCue,
		minLullPrompts:  minLullPrompts,
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
	b.WriteString("  \"continue_conversation\": <true or false>,\n")
	b.WriteString("  \"command_in_progress\": <true or false>\n")
	b.WriteString("}\n\n")
	b.WriteString("Rules:\n")
	b.WriteString("- Always produce a non-empty \"response\". Every turn gets a spoken reply — confirmations, acknowledgments, thank-yous, farewells, or even a good-natured \"sorry, didn't catch that\" for vague input. Never return an empty string.\n")
	b.WriteString("- If the utterance doesn't clearly match any command, set \"command\" to null and still say something natural in \"response\".\n")
	b.WriteString("- Set \"continue_conversation\" to true when you expect a follow-up — you asked a clarifying question or the exchange invites more back-and-forth. Set false when the exchange is clearly complete (e.g., the requested action was confirmed or the user said goodbye). Err on the side of true if unsure, so the user can chime back in.\n")
	b.WriteString("- If the utterance sounds like ambient conversation you weren't the intended recipient of, you can acknowledge it briefly (\"Sorry, were you talking to me?\" or similar) rather than going silent.\n")
	b.WriteString("- Prior turns in this conversation (if any) are included as chat history; treat them as context.\n")
	b.WriteString("- If the most recent user message is the literal marker \"(the user has gone silent)\", it means voice-command detected a lull and is asking you to generate a nudge. Produce a brief, in-character check-in that keeps things moving (e.g. if an order or task is in progress, reassure or offer something small). Don't interpret the marker as the user's actual speech or quote it back. You may set \"continue_conversation\" to false once you judge the user has truly disengaged; voice-command may override early silences up to a configured minimum, then it will honor your decision.\n")
	b.WriteString("- If a second system block titled \"Current sensor readings\" is present, use those values to inform your response when relevant (e.g. reference queue state before promising a wait time, factor in environment readings if asked). Don't quote raw JSON back at the user; translate the information into natural speech.\n")
	b.WriteString("- \"command_in_progress\" is a hard signal to voice-command about whether your most recently dispatched command is still running. Set it to true whenever a \"Current command status\" section is present and its readings show activity (queued, brewing, processing, pending, etc.). Set it to false when no status section is present, or when the status clearly indicates the command is finished, cancelled, or no command is active. While command_in_progress is true, voice-command will KEEP THE CONVERSATION OPEN regardless of continue_conversation, so the user can keep talking to you while the command runs. When the status shows the command has completed, set command_in_progress to false on the next turn so the conversation can end normally once the user's done.")
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
	s.lullCount = 0
}

// resetLullCount marks that the user has engaged (real utterance), so the
// next silence starts the nudge counter over.
func (s *service) resetLullCount() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lullCount = 0
}

// bumpLullCount increments the consecutive-silence nudge counter and
// returns the new value.
func (s *service) bumpLullCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lullCount++
	return s.lullCount
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
			// In conversation mode, a deadline-exceeded means the user went
			// silent. Instead of ending the conversation outright, ask
			// Claude to generate a nudge prompt and keep the window open.
			// If Claude judges the user has disengaged AND we've hit the
			// configured minimum number of nudges, we honor that and end.
			if inConvo && (errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "context deadline exceeded")) {
				s.handleLull(history)
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
		// The user said something real — reset the consecutive-silence
		// counter so future lulls start a fresh budget.
		s.resetLullCount()
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
			"command_in_progress", reply.CommandInProgress,
			"response", reply.Response)

		if reply.Response != "" {
			s.speak(s.workerCtx, reply.Response)
		}
		if reply.Command != "" {
			if err := s.dispatch(s.workerCtx, reply.Command); err != nil {
				s.logger.Errorw("dispatch failed", "err", err, "command", reply.Command)
			}
		}
		// command_in_progress hard-overrides continue_conversation so a
		// long-running command can't be prematurely ended mid-execution.
		keepOpen := reply.ContinueConversation || reply.CommandInProgress
		if keepOpen {
			if reply.CommandInProgress && !reply.ContinueConversation {
				s.logger.Infow("command_in_progress=true overriding continue_conversation=false")
			}
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

// silenceMarker is the synthetic user-role message voice-command injects
// into the conversation history when it detects a lull. The system prompt
// tells Claude to recognize this literal string and respond with a nudge
// rather than treat it as the user's actual words.
const silenceMarker = "(the user has gone silent)"

// handleLull runs the lull-prompt flow: ask Claude for a nudge based on
// current history, speak whatever it returns, dispatch any command it
// emits, and decide whether to keep the conversation open. If Claude sets
// continue_conversation=false and we've already produced at least
// minLullPrompts nudges, we honor that and end; otherwise we override and
// stay open for another round.
func (s *service) handleLull(history []anthropic.MessageParam) {
	s.logger.Infow("conversation lull detected; generating nudge")

	reply, newHistory, err := s.interpret(s.workerCtx, silenceMarker, history)
	if err != nil {
		// LLM roundtrip failed — fall back to the old behavior (end cue,
		// close conversation) rather than spinning on errors.
		s.logger.Errorw("lull interpret failed; ending conversation", "err", err)
		s.speakEndCue(s.workerCtx)
		s.endConversation()
		return
	}

	count := s.bumpLullCount()
	s.logger.Infow("lull decision",
		"command", reply.Command,
		"continue_conversation", reply.ContinueConversation,
		"command_in_progress", reply.CommandInProgress,
		"response", reply.Response,
		"lull_count", count,
		"min_lull_prompts", s.minLullPrompts)

	if reply.Response != "" {
		s.speak(s.workerCtx, reply.Response)
	}
	if reply.Command != "" {
		if err := s.dispatch(s.workerCtx, reply.Command); err != nil {
			s.logger.Errorw("dispatch failed during lull", "err", err, "command", reply.Command)
		}
	}

	// command_in_progress hard-overrides the end decision — if a command
	// is still running, keep the conversation alive even if Claude wants
	// to end and we've blown through the lull budget.
	if reply.CommandInProgress {
		if !reply.ContinueConversation || count >= s.minLullPrompts {
			s.logger.Infow("command_in_progress=true overriding end-of-conversation during lull")
		}
		s.extendConversation(newHistory)
		return
	}

	// Honor Claude's desire to end only once we've met the minimum budget.
	// Until then, override and keep the conversation alive.
	if !reply.ContinueConversation && count >= s.minLullPrompts {
		s.logger.Infow("ending conversation after lull budget met",
			"lull_count", count, "min_lull_prompts", s.minLullPrompts)
		s.speakEndCue(s.workerCtx)
		s.endConversation()
		return
	}
	// Either Claude wants to keep going, or we're still under the minimum.
	// Either way, extend the window and loop back to listening.
	s.extendConversation(newHistory)
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

	// Two client-side commit triggers both feed into streamCancel; the
	// Recv error path below distinguishes which one fired via the atomic
	// flags for telemetry.
	//
	// listenTimer: once we're in stateListening, the user has
	// listenTimeout of total silence before we give up.
	//
	// stableTimer: once captured content stops changing, we wait
	// stableEndpoint before committing — beats Google's ~1-2s internal
	// VAD on short utterances without closing the stream.
	var (
		listenTimer       *time.Timer
		listenTimerFired  atomic.Bool
		stableTimer       *time.Timer
		stableTimerFired  atomic.Bool
		lastStableContent string
	)
	armListenTimeout := func() {
		if listenTimer != nil {
			return
		}
		listenTimer = time.AfterFunc(s.listenTimeout, func() {
			listenTimerFired.Store(true)
			streamCancel()
		})
	}
	// resetStableTimer (re)arms the stable-endpoint timer whenever the
	// captured content changes. When content is unchanged across
	// interims, the timer keeps counting down — that's how we detect
	// "user stopped talking". Content comparison is byte-level so that
	// Google's occasional backward revisions (e.g. "expression" →
	// "espresso") also reset the timer.
	resetStableTimer := func(content string) {
		if s.stableEndpoint <= 0 || content == "" || content == lastStableContent {
			return
		}
		lastStableContent = content
		if stableTimer == nil {
			stableTimer = time.AfterFunc(s.stableEndpoint, func() {
				stableTimerFired.Store(true)
				streamCancel()
			})
			return
		}
		stableTimer.Reset(s.stableEndpoint)
	}
	if startInConversation {
		armListenTimeout()
	}

	defer func() {
		if listenTimer != nil {
			listenTimer.Stop()
		}
		if stableTimer != nil {
			stableTimer.Stop()
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
			// If this error is due to one of our client-side timers
			// cancelling streamCtx (and the outer ctx is still alive),
			// treat it as a normal commit: return whatever we captured
			// without an error.
			if streamCtx.Err() != nil && ctx.Err() == nil {
				reason := "stream cancelled"
				switch {
				case stableTimerFired.Load():
					reason = "committing utterance (stable endpoint)"
				case listenTimerFired.Load():
					reason = "listen timeout fired"
				}
				s.logger.Infow(reason,
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
					resetStableTimer(after)
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
				// fires while the user is still mid-sentence. Also (re)arm
				// the stable-endpoint timer whenever content changes.
				if post != "" {
					captureBuf.Reset()
					captureBuf.WriteString(post)
					resetStableTimer(post)
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

// interpretedReply is what the LLM is asked to return. Command is the
// Name of a configured CommandEntry, or empty / null for "no action."
// CommandInProgress is an explicit flag Claude sets when the Current
// command status block indicates the most recently dispatched command
// is still running; voice-command hard-overrides continue_conversation
// while this is true so the customer can keep talking during long
// operations like a brew.
type interpretedReply struct {
	Response             string `json:"response"`
	Command              string `json:"command"`
	ContinueConversation bool   `json:"continue_conversation"`
	CommandInProgress    bool   `json:"command_in_progress,omitempty"`
}

// fetchContext concurrently gathers sensor Readings() for the ambient
// sensors[] list and Status() for the optional command-status resource.
// Failures are captured as {"error": "..."} under their slot so the LLM
// call can still proceed with partial data.
func (s *service) fetchContext(ctx context.Context) (sensorReadings map[string]interface{}, commandStatus map[string]interface{}) {
	if len(s.sensors) == 0 && s.commandStatus == nil {
		return nil, nil
	}
	sensorReadings = make(map[string]interface{}, len(s.sensors))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, sr := range s.sensors {
		wg.Add(1)
		go func(sr sensorRef) {
			defer wg.Done()
			readings, err := sr.handle.Readings(ctx, nil)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				s.logger.Warnw("sensor readings failed", "sensor", sr.name, "err", err)
				sensorReadings[sr.name] = map[string]interface{}{"error": err.Error()}
				return
			}
			sensorReadings[sr.name] = readings
		}(sr)
	}
	if s.commandStatus != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			status, err := s.commandStatus.handle.Status(ctx)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				s.logger.Warnw("command status fetch failed", "resource", s.commandStatus.name, "err", err)
				commandStatus = map[string]interface{}{"error": err.Error()}
				return
			}
			commandStatus = status
		}()
	}
	wg.Wait()
	return sensorReadings, commandStatus
}

// sensorContextBlock formats a fresh snapshot of context as the text of
// the second (uncached) system block. Called once per interpret() turn.
// Ambient sensors[] readings are listed first, followed by the designated
// command-status resource's Status() under its own heading so Claude sees
// the latter's role distinctly.
func (s *service) sensorContextBlock(ctx context.Context) string {
	sensorReadings, commandStatus := s.fetchContext(ctx)
	var b strings.Builder
	now := time.Now().UTC().Format(time.RFC3339)

	if len(s.sensors) > 0 {
		fmt.Fprintf(&b, "Current sensor readings (as of %s):\n", now)
		for _, sr := range s.sensors {
			payload, err := json.Marshal(sensorReadings[sr.name])
			if err != nil {
				payload = []byte(fmt.Sprintf("%q", err.Error()))
			}
			fmt.Fprintf(&b, "- %q (%s): %s\n", sr.name, sr.description, payload)
		}
	}

	if s.commandStatus != nil {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		payload, err := json.Marshal(commandStatus)
		if err != nil {
			payload = []byte(fmt.Sprintf("%q", err.Error()))
		}
		desc := s.commandStatus.description
		if desc == "" {
			desc = "state of the most recently dispatched command"
		}
		fmt.Fprintf(&b, "Current command status (as of %s, from %q — %s):\n%s",
			now, s.commandStatus.name, desc, payload)
	}

	return strings.TrimRight(b.String(), "\n")
}

// interpret sends the utterance (with optional prior conversation history)
// to Claude and returns the parsed reply plus the history extended with
// this turn's user and assistant messages. Callers can thread the returned
// history back into the next interpret() call to maintain context across
// a conversation window.
func (s *service) interpret(ctx context.Context, utterance string, history []anthropic.MessageParam) (interpretedReply, []anthropic.MessageParam, error) {
	userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(utterance))
	msgs := append(append([]anthropic.MessageParam{}, history...), userMsg)

	// First block: the stable, cached instructions + command list.
	// Second block (if any sensors or a command-status sensor are
	// configured): fresh-per-turn readings, uncached, so it can change
	// between calls without invalidating the cache on the first block.
	systemBlocks := []anthropic.TextBlockParam{{
		Text:         s.systemPrompt,
		CacheControl: anthropic.NewCacheControlEphemeralParam(),
	}}
	if len(s.sensors) > 0 || s.commandStatus != nil {
		systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
			Text: s.sensorContextBlock(ctx),
		})
	}

	msg, err := s.anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     s.anthropicModel,
		MaxTokens: s.maxTokens,
		System:    systemBlocks,
		Messages:  msgs,
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
