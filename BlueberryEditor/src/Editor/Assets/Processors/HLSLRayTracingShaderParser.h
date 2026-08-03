#pragma once

#include "Blueberry\Graphics\RayTracingShader.h"

namespace Blueberry
{
	struct RayTracingShaderCompilationData
	{
		String shaderCode;

		String rayGenerationEntryPoint;
		List<String> anyHitEntryPoints;
		List<String> closestHitEntryPoints;
		List<String> missEntryPoints;
	};

	class HLSLRayTracingShaderParser
	{
	public:
		static bool Parse(const String& path, RayTracingShaderData& shaderData, RayTracingShaderCompilationData& compilationData);
	};
}