#pragma once

#include <JuceHeader.h>

class SX_OGL_Quad {
public:
	bool created = false;

	GLuint VAO, VBO, EBO;

	float left = 0;
	float top = 0;
	float width = 0.5;
	float height = 0.5;

	void create();
private:

};