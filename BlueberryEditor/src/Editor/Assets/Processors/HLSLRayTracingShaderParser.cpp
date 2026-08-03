#include "HLSLRayTracingShaderParser.h"

#include "Blueberry\Tools\FileHelper.h"

#include <regex>

namespace Blueberry
{
	bool HLSLRayTracingShaderParser::Parse(const String& path, RayTracingShaderData& shaderData, RayTracingShaderCompilationData& compilationData)
	{
		String shader = FileHelper::LoadText(path);

		std::smatch match;
		std::regex rayGenerationEntryPointRegex("#pragma\\s*raygeneration\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(shader, match, rayGenerationEntryPointRegex))
		{
			compilationData.rayGenerationEntryPoint = match[1].str();
		}

		std::regex anyHitEntryPointRegex("#pragma\\s*anyhit(\\d+)\\s*([\\w-]+)[\r?\n]");
		auto anyHitsStart = std::sregex_iterator(shader.begin(), shader.end(), anyHitEntryPointRegex);
		auto anyHitsEnd = std::sregex_iterator();

		for (std::regex_iterator i = anyHitsStart; i != anyHitsEnd; ++i)
		{
			std::smatch match = *i;
			int index = std::stoi(match[1].str());
			compilationData.anyHitEntryPoints.resize(index + 1);
			compilationData.anyHitEntryPoints[index] = String(match[2].str());
		}

		std::regex closestHitEntryPointRegex("#pragma\\s*closesthit(\\d+)\\s*([\\w-]+)[\r?\n]");
		auto closestHitsStart = std::sregex_iterator(shader.begin(), shader.end(), closestHitEntryPointRegex);
		auto closestHitsEnd = std::sregex_iterator();

		for (std::regex_iterator i = closestHitsStart; i != closestHitsEnd; ++i)
		{
			std::smatch match = *i;
			int index = std::stoi(match[1].str());
			compilationData.closestHitEntryPoints.resize(index + 1);
			compilationData.closestHitEntryPoints[index] = String(match[2].str());
		}

		std::regex missEntryPointRegex("#pragma\\s*miss(\\d+)\\s*([\\w-]+)[\r?\n]");
		auto missesStart = std::sregex_iterator(shader.begin(), shader.end(), missEntryPointRegex);
		auto missesEnd = std::sregex_iterator();

		for (std::regex_iterator i = missesStart; i != missesEnd; ++i)
		{
			std::smatch match = *i;
			int index = std::stoi(match[1].str());
			compilationData.missEntryPoints.resize(index + 1);
			compilationData.missEntryPoints[index] = String(match[2].str());
		}

		std::regex payloadSizeRegex("#pragma\\s*payload_size\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(shader, match, payloadSizeRegex))
		{
			shaderData.SetPayloadSize(static_cast<uint32_t>(std::stoi(match[1].str())));
		}

		std::regex attributesSizeRegex("#pragma\\s*attributes_size\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(shader, match, attributesSizeRegex))
		{
			shaderData.SetAttributesSize(static_cast<uint32_t>(std::stoi(match[1].str())));
		}

		std::regex rayRecursionDepthRegex("#pragma\\s*ray_recursion_depth\\s*([\\w-]+)[\r?\n]");
		if (std::regex_search(shader, match, rayRecursionDepthRegex))
		{
			shaderData.SetRayRecursionDepth(static_cast<uint32_t>(std::stoi(match[1].str())));
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