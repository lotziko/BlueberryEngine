#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\GfxShader.h"
#include "Blueberry\Graphics\ShaderOptions.h"

namespace Blueberry
{
	class GfxShader;

	class Shader : public Object
	{
		OBJECT_DECLARATION(Shader)

	public:
		Shader() = default;
		virtual ~Shader() = default;

		const ShaderOptions& GetOptions();

		void Initialize(void* vertexData, void* pixelData);
		void Initialize(void* vertexData, void* pixelData, const RawShaderOptions& options);
		static Shader* Create(void* vertexData, void* pixelData);
		static Shader* Create(void* vertexData, void* pixelData, const RawShaderOptions& options);

		static void BindProperties();

	private:
		GfxShader* m_Shader;
		ShaderOptions m_Options;

		friend struct GfxDrawingOperation;
	};
}