#include "HLSLRayTracingShaderProcessor.h"

#include "HLSLRayTracingShaderParser.h"
#include "HLSLShaderCompiler.h"

#include "Blueberry\Graphics\GraphicsAPI.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Tools\FileHelper.h"

#include <filesystem>

namespace Blueberry
{
	bool HLSLRayTracingShaderProcessor::Compile(const String& path)
	{
		if (GraphicsAPI::GetAPI() == GraphicsAPI::API::DX12)
		{
			RayTracingShaderCompilationData compilationData = {};
			if (HLSLRayTracingShaderParser::Parse(path, m_RayTracingShaderData, compilationData))
			{
				HLSLShaderCompilerDXC compiler(compilationData.shaderCode);
				if (compiler.Compile(compilationData.rayGenerationEntryPoint, compilationData.anyHitEntryPoints, compilationData.closestHitEntryPoints, compilationData.missEntryPoints, m_Shader))
				{
					return true;
				}
			}
		}
		return false;
	}

	void HLSLRayTracingShaderProcessor::Save(const String& folderPath)
	{
		std::filesystem::path path = folderPath;
		path.append("0");
		FileHelper::Save(m_Shader, StringHelper::ToString(path));
	}

	bool HLSLRayTracingShaderProcessor::Load(const String& folderPath)
	{
		std::filesystem::path path = folderPath;
		path.append("0");
		if (std::filesystem::exists(path))
		{
			m_Shader = FileHelper::LoadBinary(StringHelper::ToString(path));
			return true;
		}
		return false;
	}

	const RayTracingShaderData& HLSLRayTracingShaderProcessor::GetRayTracingShaderData()
	{
		return m_RayTracingShaderData;
	}

	const ByteData& HLSLRayTracingShaderProcessor::GetShader()
	{
		return m_Shader;
	}
}