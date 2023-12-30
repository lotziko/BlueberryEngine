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

		void Initialize(void* vertexData, void* pixelData);
		static Shader* Create(void* vertexData, void* pixelData);

		static void BindProperties();

	private:
		GfxShader* m_Shader;

		friend struct GfxDrawingOperation;
	};
}