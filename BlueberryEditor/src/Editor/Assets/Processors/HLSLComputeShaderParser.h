#pragma once

#include "Blueberry\Graphics\ComputeShader.h"

namespace Blueberry
{
	struct ComputeShaderCompilationData
	{
		String shaderCode;
		List<String> computeEntryPoints;

		DataList<KernelData> dataKernels;
	};

	class HLSLComputeShaderParser
	{
	public:
		static bool Parse(const String& path, ComputeShaderData& shaderData, ComputeShaderCompilationData& compilationData);
	};
}