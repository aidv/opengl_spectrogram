#include "OpenGL_Scene.h"

using namespace juce;
using namespace juce::gl;






struct Attributes
{
    explicit Attributes(OpenGLShaderProgram& shader)
    {
        position.reset(createAttribute(shader, "position"));
        normal.reset(createAttribute(shader, "normal"));
        sourceColour.reset(createAttribute(shader, "sourceColour"));
        textureCoordIn.reset(createAttribute(shader, "textureCoordIn"));
    }

    void enable()
    {
       
    }

    void disable()
    {
        using namespace ::juce::gl;

        if (position != nullptr)        glDisableVertexAttribArray(position->attributeID);
        if (normal != nullptr)          glDisableVertexAttribArray(normal->attributeID);
        if (sourceColour != nullptr)    glDisableVertexAttribArray(sourceColour->attributeID);
        if (textureCoordIn != nullptr)  glDisableVertexAttribArray(textureCoordIn->attributeID);
    }

    std::unique_ptr<OpenGLShaderProgram::Attribute> position, normal, sourceColour, textureCoordIn;

private:
    static OpenGLShaderProgram::Attribute* createAttribute(OpenGLShaderProgram& shader,
        const char* attributeName)
    {
        using namespace ::juce::gl;

        if (glGetAttribLocation(shader.getProgramID(), attributeName) < 0)
            return nullptr;

        return new OpenGLShaderProgram::Attribute(shader, attributeName);
    }
};

//==============================================================================
// This class just manages the uniform values that the demo shaders use.
struct Uniforms
{
    explicit Uniforms(OpenGLShaderProgram& shader)
    {
        projectionMatrix.reset(createUniform(shader, "projectionMatrix"));
        viewMatrix.reset(createUniform(shader, "viewMatrix"));
        texture.reset(createUniform(shader, "demoTexture"));
        lightPosition.reset(createUniform(shader, "lightPosition"));
        bouncingNumber.reset(createUniform(shader, "bouncingNumber"));
    }

    std::unique_ptr<OpenGLShaderProgram::Uniform> projectionMatrix, viewMatrix, texture, lightPosition, bouncingNumber;

private:
    static OpenGLShaderProgram::Uniform* createUniform(OpenGLShaderProgram& shader,
        const char* uniformName)
    {
        using namespace ::juce::gl;

        if (glGetUniformLocation(shader.getProgramID(), uniformName) < 0)
            return nullptr;

        return new OpenGLShaderProgram::Uniform(shader, uniformName);
    }
};







//==============================================================================
OpenGL_Scene::OpenGL_Scene()
{

    setOpaque(true);
    
    btn.reset (new TextButton ());
    addAndMakeVisible(btn.get());


    oglCtx.setOpenGLVersionRequired(OpenGLContext::openGL3_2);
    oglCtx.setRenderer(this);
    oglCtx.setContinuousRepainting(true);



    oglCtx.attachTo(*this);


    setSize(360, 256);
    setTopLeftPosition(64, 64);

}

OpenGL_Scene::~OpenGL_Scene()
{
    oglCtx.detach();
}

//==============================================================================





//==============================================================================
void OpenGL_Scene::paint(juce::Graphics& g)
{
}

void OpenGL_Scene::resized()
{
    //const ScopedLock lock(mutex);

    //bounds = getLocalBounds();
    //controlsOverlay->setBounds(bounds);
    //draggableOrientation.setViewport(bounds);


}




void OpenGL_Scene::newOpenGLContextCreated() {
    freeAllContextObjects();


    


    initGraphics();
}

void OpenGL_Scene::openGLContextClosing()
{
    // When the context is about to close, you must use this callback to delete
    // any GPU resources while the context is still current.
    freeAllContextObjects();

}

void OpenGL_Scene::freeAllContextObjects()
{

}



Matrix3D<float> OpenGL_Scene::getProjectionMatrix()
{
    const ScopedLock lock(mutex);

    auto w = 1.0f / (1.0f + 0.1f);
    auto h = w * bounds.toFloat().getAspectRatio(false);

    return Matrix3D<float>::fromFrustum(-w, w, -h, h, 4.0f, 30.0f);
}

Matrix3D<float> OpenGL_Scene::getViewMatrix()
{
    const ScopedLock lock(mutex);

    auto viewMatrix = Matrix3D<float>::fromTranslation({ 0.0f, 0.0f, 0.0f });// * draggableOrientation.getRotationMatrix();
    auto rotationMatrix = Matrix3D<float>::rotation({ 0.f, 0.f, 0.f });

    return viewMatrix * rotationMatrix;
}



void setShaderProgram(const String& vertexShader, const String& fragmentShader)
{
    //const ScopedLock lock(shaderMutex); // Prevent concurrent access to shader strings and status
    //newVertexShader = vertexShader;
    //newFragmentShader = fragmentShader;
}


void OpenGL_Scene::handleAsyncUpdate()
{
    const ScopedLock lock(shaderMutex); // Prevent concurrent access to shader strings and status
}




void OpenGL_Scene::configOnRender() {
    auto desktopScale = (float)oglCtx.getRenderingScale();

    int width = roundToInt(desktopScale * (float)getWidth());
    int height = roundToInt(desktopScale * (float)getHeight());

    /*glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glActiveTexture(GL_TEXTURE0);
    */

    //if (!oglCtx.isCoreProfile()) glEnable(GL_TEXTURE_2D);

    
}

void OpenGL_Scene::renderOpenGL(){

    {
        MessageManagerLock mm(Thread::getCurrentThread());
        if (!mm.lockWasGained())
            return;
    }



    //configOnRender();


    /*
    glViewport(0.0f, 0.0f, (GLsizei)getWidth(), (GLsizei)getHeight());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, getWidth(), 0, getHeight(), 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    */
    
    glUseProgram(shaderProgram);


    createQueuedQuads();


    glUseProgram(0);
}


void OpenGL_Scene::initGraphics()
{


    vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    shaderProgram = linkShaders();


    auto x = 0;


}



GLuint OpenGL_Scene::compileShader(GLenum type, const char* source) {
    auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

GLuint OpenGL_Scene::linkShaders(){
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Shaders are linked into our program and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

SX_OGL_Quad* OpenGL_Scene::createQuad(float posX, float posY, float width, float height) {
    SX_OGL_Quad* quad = new SX_OGL_Quad();

    quad->left   = mapVal(posX  , 0,  getWidth(), -1,  1);
    quad->top    = mapVal(posY  , 0, getHeight(),  1, -1);
    quad->width  = mapVal(width , 0,  getWidth(),  0,  2);
    quad->height = mapVal(height, 0, getHeight(),  0,  2);

    this->quadsToCreate.push_back(quad);

    return quad;
}



void OpenGL_Scene::removeQuad(int idx) {
    SX_OGL_Quad* quad = this->quadsCreated[idx];
    this->quadsCreated.erase(this->quadsCreated.begin() + idx);
    free(quad);
}

void OpenGL_Scene::createQueuedQuads() {
    size_t count = this->quadsToCreate.size();


    while (count > 0) {
        size_t idx = count - 1;

        SX_OGL_Quad* quad = this->quadsToCreate[idx];

        if (quad->created) continue;

        quad->create();

        glBindVertexArray(quad->VAO); // Bind the VAO containing the quad's data
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); // Draw the quad
        glBindVertexArray(0); // Unbind the VAO to clean up

        this->quadsCreated.push_back(quad);
        this->quadsToCreate.erase(this->quadsToCreate.begin() + idx);

        count--;
    }
}


GLuint OpenGL_Scene::createBuffer(float* data, GLsizeiptr dataSize) {
    GLuint vbo;
    glGenBuffers(1, &vbo); // Generate a buffer ID
    glBindBuffer(GL_ARRAY_BUFFER, vbo); // Bind the buffer as a vertex buffer


    glBufferData(GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);

    return vbo;
}


GLuint OpenGL_Scene::createUniform(const GLchar* uniformName) {
    glUseProgram(shaderProgram);
    return glGetUniformLocation(shaderProgram, uniformName);

}

GLuint OpenGL_Scene::createTexture(int width, int height, float* data, const GLuint uniformID) {


    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters as needed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Assuming your image data is suitable for creating a texture directly
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture


    glActiveTexture(GL_TEXTURE0); // Activate texture unit 0
    glBindTexture(GL_TEXTURE_2D, textureID); // Bind your texture
    glUniform1i(uniformID, 0); // Tell the shader the texture is on unit 0

    return textureID;
}

void updateBuffer(GLuint textureID, float* newData, GLsizeiptr dataSize, unsigned int offset, int newWidth, int newHeight){
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, newWidth, newHeight, GL_RGB, GL_FLOAT, newData);
    glBindTexture(GL_TEXTURE_2D, 0);
}







/***************/

float OpenGL_Scene::mapVal(float value, float aMin, float aMax, float bMin, float bMax) {
    //return value;
    return (value - aMin) * (bMax - bMin) / (aMax - aMin) + bMin;
}