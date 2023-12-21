#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\GfxShader.h"

namespace Blueberry
{
	class GfxShader;

	class Shader : public Object
	{
		OBJECT_DECLARATION(Shader)

	public:
		Shader() = default;
		Shader(std::wstring shaderPath);

		static void BindProperties();

	private:
		GfxShader* m_Shader;

		friend struct GfxDrawingOperation;
	};
}