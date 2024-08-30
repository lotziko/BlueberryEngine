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

			std::vector<std::string> vertexKeywords;
			std::vector<std::string> fragmentKeywords;
		};

		std::vector<Pass> passes;
	};

	class HLSLShaderParser
	{
	public:
		static bool Parse(const std::string& path, ShaderData& shaderData, ShaderCompilationData& compilationData);
	};
}