#pragma once

#include <JuceHeader.h>

#include "./ui/opengl_scene/opengl_scene.h"
#include "./ui/spectrogram_viewer/spectrogram_viewer.h"

//==============================================================================
/*
    This component lives inside our window, and this is where you should put all
    your controls and content.
*/
class MainComponent  : public juce::Component
{
public:
    //==============================================================================
    MainComponent();
    ~MainComponent() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;



    StringArray getRenderingEngines() { return renderingEngines; }
    int getCurrentRenderingEngine() { return currentRenderingEngineIdx; }
    void setRenderingEngine(int index);




    Spectrogram_Viewer specView;

    TextButton resizeBtn;
    TextButton createQuadsBtn;


private:
    void parentHierarchyChanged() override;
    void updateRenderingEngine(int index);
    

    std::unique_ptr<Component> contentComponent;



    OpenGLContext openGLContext;
    ComponentPeer* peer = nullptr;
    StringArray renderingEngines;
    int currentRenderingEngineIdx = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
