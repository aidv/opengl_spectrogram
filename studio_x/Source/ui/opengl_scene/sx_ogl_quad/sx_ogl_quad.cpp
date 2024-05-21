#include "sx_ogl_quad.h"

using namespace juce::gl;

void SX_OGL_Quad::create() {
    if (created) return;

    /*GLfloat vertices[] = {
        // positions        
         0.8f,  1.f, 0.0f,  // top right
         1.f, -1.f, 0.0f,  // bottom right
        -1.f, -1.f, 0.0f,  // bottom left
        -1.f,  1.f, 0.0f   // top left 
    };*/

    GLfloat vertices[] = {
        // positions        
        left + width, top - height, 0.f,  // top right
        left + width, top, 0.f, // bottom right
        left        , top         , 0.f,  // bottom left
        left        , top - height , 0.f   // top left 
    };

    unsigned int indices[] = {
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };



    // Generate and bind the Vertex Array Object
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Generate and bind the Vertex Buffer Object, upload the vertex data
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Generate and bind the Element Buffer Object, upload the indices
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Set the vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), NULL);
    glEnableVertexAttribArray(0);

    // Unbind the VAO (it's always a good practice to unbind any buffer/array to prevent strange bugs)
    glBindVertexArray(0);

    created = true;
}