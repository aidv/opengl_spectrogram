#pragma once

#include <JuceHeader.h>
#include "./sx_ogl_quad/sx_ogl_quad.h"

using namespace juce;
//==============================================================================
/*
    This component lives inside our window, and this is where you should put all
    your controls and content.
*/
class OpenGL_Scene : public juce::Component, private juce::OpenGLRenderer, private juce::AsyncUpdater
{
public:
    //==============================================================================
    OpenGL_Scene();
    ~OpenGL_Scene() override;


    //==============================================================================
    void paint(juce::Graphics&) override;

    void resized() override;


    OpenGLContext oglCtx;
    std::unique_ptr<TextButton> btn;

    void newOpenGLContextCreated() override;
    void openGLContextClosing() override;
    void freeAllContextObjects();


    void configOnRender();
    void renderOpenGL() override;


    Matrix3D<float> getProjectionMatrix();
    Matrix3D<float> getViewMatrix();
    CriticalSection mutex;
    Rectangle<int> bounds;


    std::unique_ptr<OpenGLShaderProgram> shader;
    CriticalSection shaderMutex;
    void setShaderProgram(const String& vertexShader, const String& fragmentShader);

    void initGraphics();
    
    GLuint compileShader(GLenum type, const char* source);
    GLuint linkShaders();


    GLuint createBuffer(float* data, GLsizeiptr dataSize);
    GLuint createUniform(const GLchar* uniformName);
    GLuint createTexture(int width, int height, float* data, const GLuint uniformID);


    std::vector<SX_OGL_Quad*> quadsToCreate;
    std::vector<SX_OGL_Quad*> quadsCreated;
    SX_OGL_Quad* createQuad(float posX, float posY, float width, float height);
    SX_OGL_Quad* init(SX_OGL_Quad *quad, float posX, float posY, float width, float height);

    void removeQuad(int idx);
    void createQueuedQuads();



    /*********/

    float mapVal(float value, float aMin, float aMax, float bMin, float bMax);
private:
    //==============================================================================
    // Your private member variables go here...


    void handleAsyncUpdate() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OpenGL_Scene)




    GLuint shaderProgram;

    GLuint vertexShader;
    GLuint fragmentShader;

    const char* vertexShaderSource = R"GLSL(
    #version 330 core
    layout (location = 0) in vec3 position;

    attribute vec2 quadCoordIn;
    varying vec2 quadCoord;


    void main() {
        quadCoord = quadCoordIn;
        gl_Position = vec4(position, 1.0);
    }
)GLSL";

    const char* fragmentShaderSource = R"GLSL(
    #version 330 core
    out vec4 FragColor;
    void main() {

        FragColor = vec4(1.0, .5, 0.2, 1.0);
        //FragColor = (gl_FragCoord.x<1.0) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
        //FragColor = vec4(1.0, gl_Position.x, 0.2, 1.0);
    }
)GLSL";
};
