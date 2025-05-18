#pragma once

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	struct ShaderCompilationData
	{
		struct Pass
		{
			String shaderCode;

			String vertexEntryPoint;
			String geometryEntryPoint;
			String fragmentEntryPoint;

			List<String> vertexKeywords;
			List<String> fragmentKeywords;

			List<String> globalVertexKeywords;
			List<String> globalFragmentKeywords;
		};

		List<Pass> passes;
		DataList<PassData> dataPasses;
	};

	class HLSLShaderParser
	{
	public:
		static bool Parse(const String& path, ShaderData& shaderData, ShaderCompilationData& compilationData);
	};
}