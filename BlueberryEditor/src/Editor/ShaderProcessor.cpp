#include "bbpch.h"
#include "ShaderProcessor.h"
#include "Blueberry\Graphics\ShaderOptions.h"
#include "Blueberry\Tools\StringConverter.h"
#include "Blueberry\Tools\StringHelper.h"
#include "Blueberry\Tools\FileHelper.h"
#include <filesystem>
#include <fstream>

namespace Blueberry
{
	HRESULT ShaderProcessorInclude::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
	{
		std::string filePath("assets/" + std::string(pFileName));
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

	HRESULT ShaderProcessorInclude::Close(LPCVOID pData)
	{
		std::free(const_cast<void*>(pData));
		return S_OK;
	}

	void ShaderProcessor::Process(const std::string& path, std::string& shaderCode, RawShaderOptions& options)
	{
		std::string shaderData;
		FileHelper::Load(shaderData, path);

		if (shaderData.find("Shader") == 0)
		{
			auto shaderBeginTag = "HLSLBEGIN";
			auto shaderEndTag = "HLSLEND";
			size_t shaderBegin;
			size_t shaderEnd;

			if ((shaderBegin = shaderData.find(shaderBeginTag)) == std::string::npos || (shaderEnd = shaderData.find(shaderEndTag, shaderBegin)) == std::string::npos)
			{
				return;
			}
			shaderBegin += sizeof(shaderBeginTag) + 1;
			shaderCode = shaderData.substr(shaderBegin, shaderEnd - shaderBegin);

			size_t optionsTagBegin;
			if ((optionsTagBegin = shaderData.find("Options")) != std::string::npos)
			{
				auto optionsBeginTag = "{";
				auto optionsEndTag = "}";

				size_t optionsBegin;
				size_t optionsEnd;
				if ((optionsBegin = shaderData.find(optionsBeginTag, optionsTagBegin)) != std::string::npos && (optionsEnd = shaderData.find(optionsEndTag, optionsBegin)) != std::string::npos)
				{
					optionsBegin += 3;
					optionsEnd -= 2;

					for (size_t i = optionsBegin; i < optionsEnd; ++i)
					{
						if (shaderData[i] < 32)
						{
							continue;
						}
						else
						{
							std::string optionKey;
							for (size_t j = i; j < optionsEnd; ++j)
							{
								if (shaderData[j] == ' ')
								{
									optionKey = shaderData.substr(i, j - i);
									i = j + 1;
									break;
								}
							}
							std::string optionValue;
							for (size_t j = i; j < optionsEnd; ++j)
							{
								if (shaderData[j] < 32)
								{
									optionValue = shaderData.substr(i, j - i);
									i = j + 1;
									break;
								}
							}
							options.insert_or_assign(optionKey, optionValue);
						}
					}
				}
			}
		}
		else
		{
			shaderCode = shaderData;
		}
	}

	void* ShaderProcessor::Compile(const std::string& shaderData, const char* entryPoint, const char* model, const std::string& blobPath)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
		
		ShaderProcessorInclude include;
		ComPtr<ID3DBlob> shader;
		ComPtr<ID3DBlob> error;

		HRESULT hr = D3DCompile2(shaderData.data(), shaderData.size(), nullptr, nullptr, &include, entryPoint, model, flags, 0, 0, nullptr, 0, shader.GetAddressOf(), error.GetAddressOf()); //D3DCompileFromFile(StringConverter::StringToWide(path).c_str(), nullptr, &include, entryPoint, model, flags, 0, shader.GetAddressOf(), error.GetAddressOf());
		
		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader.");
			BB_ERROR((char*)error->GetBufferPointer());
			error->Release();
			return nullptr;
		}
		if (blobPath.length() > 0)
		{
			D3DWriteBlobToFile(shader.Get(), StringConverter::StringToWide(blobPath).c_str(), true);
		}
		return shader.Detach();
	}

	void* ShaderProcessor::Load(const std::string& path)
	{
		ComPtr<ID3DBlob> shader;
		HRESULT hr = D3DReadFileToBlob(StringConverter::StringToWide(path).c_str(), shader.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to load shader: " + std::string(path.begin(), path.end()));
			return nullptr;
		}
		return shader.Detach();
	}
}
