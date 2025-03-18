#pragma once

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	struct ShaderCompilationData
	{
		struct Pass
		{
			std::string shaderCode;

			std::string vertexEntryPoint;
			std::string geometryEntryPoint;
			std::string fragmentEntryPoint;

			List<std::string> vertexKeywords;
			List<std::string> fragmentKeywords;

			List<std::string> globalVertexKeywords;
			List<std::string> globalFragmentKeywords;
		};

		List<Pass> passes;
		List<PassData*> dataPasses;
	};

	class HLSLShaderParser
	{
	public:
		static bool Parse(const std::string& path, ShaderData& shaderData, ShaderCompilationData& compilationData);
	};
}