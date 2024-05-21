#pragma once

#include "../opengl_scene/opengl_scene.h"
#include <JuceHeader.h>

using namespace juce;

class Spectrogram_Viewer final : public OpenGL_Scene
{
public:
    //==============================================================================
    Spectrogram_Viewer();
    ~Spectrogram_Viewer() override;

    void generateSnapshot();
private:
   
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Spectrogram_Viewer)
};
