#include "bbpch.h"
#include "HLSLShaderProcessor.h"

#include "Blueberry\Tools\StringConverter.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Tools\FileHelper.h"

#include "HLSLShaderParser.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	HRESULT HLSLShaderProcessorInclude::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
	{
		std::string filePath("assets/shaders/" + std::string(pFileName));
		if (!std::filesystem::exists(filePath))
		{
			return E_FAIL;
		}

		// Based on https://github.com/holy-shit/clion-directx-example/blob/master/main.cpp
		UINT dataSize;
		char* buffer;

		std::ifstream infile;
		infile.open(filePath, std::ios::binary);
		infile.seekg(0, std::ios::end);
		dataSize = infile.tellg();
		infile.seekg(0, std::ios::beg);
		buffer = (char*)malloc(dataSize);
		infile.read(buffer, dataSize);
		infile.close();

		*pBytes = dataSize;
		*ppData = buffer;

		return S_OK;
	}

	HRESULT HLSLShaderProcessorInclude::Close(LPCVOID pData)
	{
		std::free(const_cast<void*>(pData));
		return S_OK;
	}

	HLSLShaderProcessor::~HLSLShaderProcessor()
	{
		for (auto& blob : m_Blobs)
		{
			blob.Reset();
		}
		m_Blobs.clear();
	}

	bool HLSLShaderProcessor::Compile(const std::string& path)
	{
		ShaderCompilationData compilationData = {};
		if (HLSLShaderParser::Parse(path, m_ShaderData, compilationData))
		{
			for (int i = 0; i < compilationData.passes.size(); ++i)
			{
				auto& compilationPass = compilationData.passes[i];
				PassData* pass = compilationData.dataPasses[i];
				size_t vertexVariantCount = Max(pow(2, compilationPass.vertexKeywords.size()), 1);
				size_t fragmentVariantCount = Max(pow(2, compilationPass.fragmentKeywords.size()), 1);

				if (!compilationPass.vertexEntryPoint.empty())
				{
					int keywordCount = compilationPass.vertexKeywords.size();
					D3D_SHADER_MACRO keywords[256];
					for (int i = 0; i < keywordCount; ++i)
					{
						keywords[i].Name = compilationPass.vertexKeywords[i].c_str();
					}

					keywords[keywordCount].Name = nullptr;
					keywords[keywordCount].Definition = nullptr;

					pass->SetVertexOffset(m_VariantsData.vertexShaderIndices.size());

					for (size_t i = 0; i < vertexVariantCount; ++i)
					{
						for (int j = 0; j < keywordCount; ++j)
						{
							keywords[j].Definition = i & j ? "1" : "0";
						}

						ComPtr<ID3DBlob> vertexBlob;
						if (!Compile(compilationPass.shaderCode, compilationPass.vertexEntryPoint.c_str(), "vs_5_0", keywords, vertexBlob))
						{
							return false;
						}
						m_VariantsData.shaders.emplace_back(vertexBlob.Get());
						m_VariantsData.vertexShaderIndices.emplace_back(m_Blobs.size());
						m_Blobs.emplace_back(vertexBlob);
					}
				}
				else
				{
					return false;
				}

				if (!compilationPass.geometryEntryPoint.empty())
				{
					pass->SetGeometryOffset(m_VariantsData.geometryShaderIndices.size());

					ComPtr<ID3DBlob> geometryBlob;
					if (!Compile(compilationPass.shaderCode, compilationPass.geometryEntryPoint.c_str(), "gs_5_0", nullptr, geometryBlob))
					{
						return false;
					}
					m_VariantsData.shaders.emplace_back(geometryBlob.Get());
					m_VariantsData.geometryShaderIndices.emplace_back(m_Blobs.size());
					m_Blobs.emplace_back(geometryBlob);
				}
				else
				{
					m_VariantsData.geometryShaderIndices.emplace_back(-1);
				}

				if (!compilationPass.fragmentEntryPoint.empty())
				{
					int keywordCount = compilationPass.fragmentKeywords.size();
					D3D_SHADER_MACRO keywords[256];
					for (int i = 0; i < keywordCount; ++i)
					{
						keywords[i].Name = compilationPass.fragmentKeywords[i].c_str();
					}

					keywords[keywordCount].Name = nullptr;
					keywords[keywordCount].Definition = nullptr;

					pass->SetFragmentOffset(m_VariantsData.fragmentShaderIndices.size());

					for (size_t i = 0; i < fragmentVariantCount; ++i)
					{
						for (int j = 0; j < keywordCount; ++j)
						{
							keywords[j].Definition = i & (j + 1) ? "1" : "0";
						}

						ComPtr<ID3DBlob> fragmentBlob;
						if (!Compile(compilationPass.shaderCode, compilationPass.fragmentEntryPoint.c_str(), "ps_5_0", keywords, fragmentBlob))
						{
							return false;
						}
						m_VariantsData.shaders.emplace_back(fragmentBlob.Get());
						m_VariantsData.fragmentShaderIndices.emplace_back(m_Blobs.size());
						m_Blobs.emplace_back(fragmentBlob);
					}
				}
				else
				{
					return false;
				}
			}
			m_ShaderData.SetPasses(compilationData.dataPasses);
		}
		return true;
	}

	void HLSLShaderProcessor::SaveVariants(const std::string& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		UINT vertexShaderCount = (UINT)m_VariantsData.vertexShaderIndices.size();
		UINT geometryShaderCount = (UINT)m_VariantsData.geometryShaderIndices.size();
		UINT fragmentShaderCount = (UINT)m_VariantsData.fragmentShaderIndices.size();
		UINT blobsCount = (UINT)m_Blobs.size();
		std::ofstream output;
		output.open(indexesPath, std::ofstream::binary);
		output.write((char*)&vertexShaderCount, sizeof(UINT));
		output.write((char*)m_VariantsData.vertexShaderIndices.data(), sizeof(UINT) * vertexShaderCount);
		output.write((char*)&geometryShaderCount, sizeof(UINT));
		output.write((char*)m_VariantsData.geometryShaderIndices.data(), sizeof(UINT) * geometryShaderCount);
		output.write((char*)&fragmentShaderCount, sizeof(UINT));
		output.write((char*)m_VariantsData.fragmentShaderIndices.data(), sizeof(UINT) * fragmentShaderCount);
		output.write((char*)&blobsCount, sizeof(UINT));
		output.close();

		for (size_t i = 0; i < m_Blobs.size(); ++i)
		{
			std::filesystem::path path = folderPath;
			path.append(std::to_string(i));
			ComPtr<ID3DBlob> blob = m_Blobs[i];
			if (blob->GetBufferSize() > 0)
			{
				D3DWriteBlobToFile(blob.Get(), StringConverter::StringToWide(path.string()).c_str(), true);
			}
		}
	}

	bool HLSLShaderProcessor::LoadVariants(const std::string& folderPath)
	{
		std::filesystem::path indexesPath = folderPath;
		indexesPath.append("indexes");

		if (std::filesystem::exists(indexesPath))
		{
			UINT vertexShaderCount;
			UINT geometryShaderCount;
			UINT fragmentShaderCount;
			UINT blobsCount;
			std::ifstream input;
			input.open(indexesPath, std::ofstream::binary);
			input.read((char*)&vertexShaderCount, sizeof(UINT));
			m_VariantsData.vertexShaderIndices.resize(vertexShaderCount);
			input.read((char*)m_VariantsData.vertexShaderIndices.data(), sizeof(UINT) * vertexShaderCount);
			input.read((char*)&geometryShaderCount, sizeof(UINT));
			m_VariantsData.geometryShaderIndices.resize(geometryShaderCount);
			input.read((char*)m_VariantsData.geometryShaderIndices.data(), sizeof(UINT) * geometryShaderCount);
			input.read((char*)&fragmentShaderCount, sizeof(UINT));
			m_VariantsData.fragmentShaderIndices.resize(fragmentShaderCount);
			input.read((char*)m_VariantsData.fragmentShaderIndices.data(), sizeof(UINT) * fragmentShaderCount);
			input.read((char*)&blobsCount, sizeof(UINT));
			input.close();

			for (UINT i = 0; i < blobsCount; ++i)
			{
				ComPtr<ID3DBlob> blob;
				std::filesystem::path path = folderPath;
				path.append(std::to_string(i));
				std::string stringPath = path.string();
				HRESULT hr = D3DReadFileToBlob(StringConverter::StringToWide(stringPath).c_str(), blob.GetAddressOf());
				if (FAILED(hr))
				{
					BB_ERROR("Failed to load shader: " + std::string(stringPath.begin(), stringPath.end()));
					return false;
				}
				m_Blobs.emplace_back(blob);
				m_VariantsData.shaders.emplace_back(blob.Get());
			}
			return true;
		}
		return false;
	}

	const ShaderData& HLSLShaderProcessor::GetShaderData()
	{
		return m_ShaderData;
	}

	const VariantsData& HLSLShaderProcessor::GetVariantsData()
	{
		return m_VariantsData;
	}

	bool HLSLShaderProcessor::Compile(const std::string& shaderCode, const char* entryPoint, const char* model, D3D_SHADER_MACRO* keywords, ComPtr<ID3DBlob>& blob)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		HLSLShaderProcessorInclude include = {};
		ComPtr<ID3DBlob> temporaryBlob;
		ComPtr<ID3DBlob> error;

		HRESULT hr = D3DCompile2(shaderCode.data(), shaderCode.size(), nullptr, keywords, &include, entryPoint, model, flags, 0, 0, nullptr, 0, temporaryBlob.GetAddressOf(), error.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader.");
			BB_ERROR((char*)error->GetBufferPointer());
			error->Release();
			return false;
		}

		hr = D3DStripShader(temporaryBlob->GetBufferPointer(), temporaryBlob->GetBufferSize(), D3DCOMPILER_STRIP_DEBUG_INFO | D3DCOMPILER_STRIP_TEST_BLOBS, blob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to strip shader.");
			return false;
		}

		return true;
	}
}
