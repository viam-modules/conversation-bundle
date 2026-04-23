package main

import (
	texttospeech "conversationbundle/resources/text-to-speech"
	voicecommand "conversationbundle/resources/voice-command"

	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
)

func main() {
	// ModularMain can take multiple APIModel arguments, if your module implements multiple models.
	module.ModularMain(
		resource.APIModel{API: generic.API, Model: texttospeech.Model},
		resource.APIModel{API: generic.API, Model: voicecommand.Model},
	)
}
