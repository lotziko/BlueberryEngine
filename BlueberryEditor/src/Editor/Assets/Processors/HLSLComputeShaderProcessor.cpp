#include "HLSLComputeShaderProcessor.h"

#include "HLSLComputeShaderParser.h"
#include "HLSLShaderProcessor.h"

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
			for (int i = 0; i < compilationData.computeEntryPoints.size(); ++i)
			{
				ComPtr<ID3DBlob> computeBlob;
				if (!Compile(compilationData.shaderCode, compilationData.computeEntryPoints[i].c_str(), "cs_5_0", computeBlob))
				{
					return false;
				}
				ByteData data(computeBlob->GetBufferSize());
				memcpy(data.data(), computeBlob->GetBufferPointer(), computeBlob->GetBufferSize());
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

	bool HLSLComputeShaderProcessor::Compile(const String& shaderCode, const char* entryPoint, const char* model, ComPtr<ID3DBlob>& blob)
	{
		uint32_t flags = D3DCOMPILE_ENABLE_STRICTNESS;

		HLSLShaderProcessorInclude include = {};
		ComPtr<ID3DBlob> temporaryBlob;
		ComPtr<ID3DBlob> error;

		HRESULT hr = D3DCompile2(shaderCode.data(), shaderCode.size(), nullptr, nullptr, &include, entryPoint, model, flags, 0, 0, nullptr, 0, temporaryBlob.GetAddressOf(), error.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile compute shader.");
			BB_ERROR(static_cast<char*>(error->GetBufferPointer()));
			error->Release();
			return false;
		}

		hr = D3DStripShader(temporaryBlob->GetBufferPointer(), temporaryBlob->GetBufferSize(), D3DCOMPILER_STRIP_DEBUG_INFO | D3DCOMPILER_STRIP_TEST_BLOBS, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to strip compute shader.");
			return false;
		}

		return true;
	}
}
