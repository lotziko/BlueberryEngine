#include "HLSLComputeShaderProcessor.h"

#include "HLSLComputeShaderParser.h"
#include "HLSLShaderCompiler.h"

#include "Blueberry\Graphics\GraphicsAPI.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Tools\FileHelper.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	bool HLSLComputeShaderProcessor::Compile(const String& path)
	{
		ComputeShaderCompilationData compilationData = {};
		if (HLSLComputeShaderParser::Parse(path, m_ComputeShaderData, compilationData))
		{
			std::unique_ptr<HLSLShaderCompiler> compiler;
			if (GraphicsAPI::GetAPI() == GraphicsAPI::API::DX11)
			{
				compiler = std::make_unique<HLSLShaderCompilerFXC>(compilationData.shaderCode);
			}
			else
			{
				compiler = std::make_unique<HLSLShaderCompilerDXC>(compilationData.shaderCode);
			}

			for (int i = 0; i < compilationData.computeEntryPoints.size(); ++i)
			{
				ByteData data;
				if (!compiler->Compile(compilationData.computeEntryPoints[i], HLSLShaderCompilerProfile::Compute, 0, data))
				{
					return false;
				}
				m_Shaders.push_back(std::move(data));
			}
			m_ComputeShaderData.SetKernels(compilationData.dataKernels);
		}
		else
		{
			return false;
		}
		return true;
	}

	void HLSLComputeShaderProcessor::SaveKernels(const String& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		uint32_t blobsCount = static_cast<uint32_t>(m_Shaders.size());
		std::ofstream output;
		output.open(indexesPath, std::ofstream::binary);
		output.write(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));
		output.close();

		for (size_t i = 0; i < m_Shaders.size(); ++i)
		{
			std::filesystem::path path = folderPath;
			path.append(std::to_string(i));
			FileHelper::Save(m_Shaders[i], StringHelper::ToString(path));
		}
	}

	bool HLSLComputeShaderProcessor::LoadKernels(const String& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		if (std::filesystem::exists(indexesPath))
		{
			uint32_t blobsCount;
			std::ifstream input;
			input.open(indexesPath, std::ofstream::binary);
			input.read(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));
			input.close();

			for (uint32_t i = 0; i < blobsCount; ++i)
			{
				ComPtr<ID3DBlob> blob;
				std::filesystem::path path = folderPath;
				path.append(std::to_string(i));
				String stringPath = StringHelper::ToString(path);
				if (!std::filesystem::exists(path))
				{
					BB_ERROR("Failed to load shader: " << stringPath);
					return false;
				}
				ByteData data = FileHelper::LoadBinary(stringPath);
				m_Shaders.push_back(std::move(data));
			}
			return true;
		}
		return false;
	}

	const ComputeShaderData& HLSLComputeShaderProcessor::GetComputeShaderData()
	{
		return m_ComputeShaderData;
	}

	const List<ByteData>& HLSLComputeShaderProcessor::GetShaders()
	{
		return m_Shaders;
	}
}
