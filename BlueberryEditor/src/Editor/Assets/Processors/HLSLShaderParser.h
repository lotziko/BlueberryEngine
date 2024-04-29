#pragma once

#include "Blueberry\Graphics\ShaderOptions.h"

namespace Blueberry
{
	class HLSLShaderParser
	{
	public:
		static bool Parse(const std::string& path, std::string& shaderCode, RawShaderOptions& options);
	};
}