#include "HLSLComputeShaderProcessor.h"

#include "HLSLComputeShaderParser.h"
#include "HLSLShaderProcessor.h"

#include "Blueberry\Tools\StringConverter.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	HLSLComputeShaderProcessor::~HLSLComputeShaderProcessor()
	{
		for (auto& blob : m_Blobs)
		{
			blob.Reset();
		}
		m_Blobs.clear();
	}

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
					m_Blobs.emplace_back(nullptr);
					m_Shaders.emplace_back(nullptr);
					continue;
				}
				m_Blobs.emplace_back(computeBlob);
				m_Shaders.emplace_back(computeBlob.Get());
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

		uint32_t blobsCount = static_cast<uint32_t>(m_Blobs.size());
		std::ofstream output;
		output.open(indexesPath, std::ofstream::binary);
		output.write(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));
		output.close();

		for (size_t i = 0; i < m_Blobs.size(); ++i)
		{
			std::filesystem::path path = folderPath;
			path.append(std::to_string(i));
			ComPtr<ID3DBlob> blob = m_Blobs[i];
			if (blob->GetBufferSize() > 0)
			{
				D3DWriteBlobToFile(blob.Get(), StringConverter::StringToWide(String(path.string())).c_str(), true);
			}
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
				String stringPath = String(path.string());
				HRESULT hr = D3DReadFileToBlob(StringConverter::StringToWide(stringPath).c_str(), blob.GetAddressOf());
				if (FAILED(hr))
				{
					BB_ERROR("Failed to load shader: " + String(stringPath.begin(), stringPath.end()));
					return false;
				}
				m_Blobs.emplace_back(blob);
				m_Shaders.emplace_back(blob.Get());
			}
			return true;
		}
		return false;
	}

	const ComputeShaderData& HLSLComputeShaderProcessor::GetComputeShaderData()
	{
		return m_ComputeShaderData;
	}

	const List<void*>& HLSLComputeShaderProcessor::GetShaders()
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
