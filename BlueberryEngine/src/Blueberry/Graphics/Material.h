#pragma once

class Shader;

class Material : public Object
{
private:
	Shader* m_Shader;
};