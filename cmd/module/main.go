package main

import (
	texttospeech "conversationbundle/resources/text-to-speech"
	voicecommand "conversationbundle/resources/voice-command"
	weathersensor "conversationbundle/resources/weather-sensor"

	"go.viam.com/rdk/components/sensor"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
)

func main() {
	// ModularMain can take multiple APIModel arguments, if your module implements multiple models.
	module.ModularMain(
		resource.APIModel{API: generic.API, Model: texttospeech.Model},
		resource.APIModel{API: generic.API, Model: voicecommand.Model},
		resource.APIModel{API: sensor.API, Model: weathersensor.Model},
	)
}
