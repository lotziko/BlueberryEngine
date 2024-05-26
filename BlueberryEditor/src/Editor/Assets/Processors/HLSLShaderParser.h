#pragma once

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	class HLSLShaderParser
	{
	public:
		static bool Parse(const std::string& path, std::string& shaderCode, ShaderData& data);
	};
}