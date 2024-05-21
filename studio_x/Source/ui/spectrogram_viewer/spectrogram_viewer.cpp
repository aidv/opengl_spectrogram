#include "Spectrogram_Viewer.h"

using namespace juce;
using namespace juce::gl;



Spectrogram_Viewer::Spectrogram_Viewer(){

}

Spectrogram_Viewer::~Spectrogram_Viewer(){
}


void Spectrogram_Viewer::generateSnapshot(){
	DBG("generating snapshot");

	for (int i = 0; i < 4; i++){
		createQuad(32 * i, 32 * i, 32, 32);
	}


	/*
	createQuad(0, 0, 0.1, 0.1); //center
	createQuad(-1, -1, 0.1, 0.1); //bottom left
	createQuad(0.9, -1, 0.1, 0.1); //bottom right
	createQuad(0.9, 0.9, 0.1, 0.1); //top right
	createQuad(-1, 0.9, 0.1, 0.1); //top left
	*/
}