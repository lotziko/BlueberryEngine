#include "HLSLComputeShaderParser.h"

#include "Blueberry\Tools\FileHelper.h"

#include <regex>

namespace Blueberry
{
	bool HLSLComputeShaderParser::Parse(const String& path, ComputeShaderData& shaderData, ComputeShaderCompilationData& compilationData)
	{
		String shader;
		FileHelper::Load(shader, path);

		std::regex variantRegex("#pragma\\s*compute\\s*([\\w-]+)[\r?\n]");
		auto variantsStart = std::sregex_iterator(shader.begin(), shader.end(), variantRegex);
		auto variantsEnd = std::sregex_iterator();

		for (std::regex_iterator i = variantsStart; i != variantsEnd; ++i)
		{
			std::smatch match = *i;
			String name = String(match[1].str());

			KernelData kernelData = {};
			kernelData.SetName(name);

			compilationData.computeEntryPoints.emplace_back(name);
			compilationData.dataKernels.emplace_back(kernelData);
		}

		size_t offset = 0;
		while ((offset = shader.find("#pragma")) != String::npos)
		{
			size_t end = shader.find("\n", offset);
			shader.replace(offset, end - offset, " ");
		}
		compilationData.shaderCode = shader;

		return true;
	}
}