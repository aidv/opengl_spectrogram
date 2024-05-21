#include "MainComponent.h"
#include "./ui/opengl_scene/opengl_scene.h"

using namespace juce;



//==============================================================================
MainComponent::MainComponent()
{
    setSize(600, 400);

    addAndMakeVisible(contentComponent.get());

    addAndMakeVisible(specView);

    addAndMakeVisible(resizeBtn);
    resizeBtn.setButtonText("Resize");
    resizeBtn.setSize(64, 32);
    resizeBtn.onClick = [&]() {
        specView.setSize(480, 640);
    };




    addAndMakeVisible(createQuadsBtn);
    createQuadsBtn.setButtonText("Create Quads");
    createQuadsBtn.setSize(64, 32);
    createQuadsBtn.setTopLeftPosition(0, 34);
    createQuadsBtn.onClick = [&]() {
        specView.generateSnapshot();
    };




    specView.generateSnapshot();
}

MainComponent::~MainComponent()
{
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setFont (juce::Font (16.0f));
    g.setColour (juce::Colours::white);
    g.drawText ("Hello World!", getLocalBounds(), juce::Justification::centred, true);
}

void MainComponent::resized()
{
    // This is called when the MainComponent is resized.23
    // If you add any child components, this is where you should
    // update their positions.
}

void MainComponent::setRenderingEngine(int renderingEngineIndex)
{
    if (renderingEngineIndex != currentRenderingEngineIdx)
        updateRenderingEngine(renderingEngineIndex);
}

void MainComponent::parentHierarchyChanged()
{
    auto* newPeer = getPeer();

    if (peer != newPeer)
    {
        peer = newPeer;

        auto previousRenderingEngine = renderingEngines[currentRenderingEngineIdx];

        renderingEngines.clear();
        if (peer != nullptr)
            renderingEngines = peer->getAvailableRenderingEngines();

        renderingEngines.add("OpenGL Renderer");

        currentRenderingEngineIdx = renderingEngines.indexOf(previousRenderingEngine);

        if (currentRenderingEngineIdx < 0)
        {
#if JUCE_ANDROID
            currentRenderingEngineIdx = (renderingEngines.size() - 1);
#else
            currentRenderingEngineIdx = peer->getCurrentRenderingEngine();
#endif
        }

        updateRenderingEngine(currentRenderingEngineIdx);
    }
}

void MainComponent::updateRenderingEngine(int renderingEngineIndex)
{
    if (renderingEngineIndex == (renderingEngines.size() - 1))
    {
        
        openGLContext.attachTo(*getTopLevelComponent());
    }
    else
    {
        openGLContext.detach();
        peer->setCurrentRenderingEngine(renderingEngineIndex);
    }

    currentRenderingEngineIdx = renderingEngineIndex;
}

