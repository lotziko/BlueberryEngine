#include "bbpch.h"
#include "ShaderProcessor.h"
#include "Blueberry\Tools\StringConverter.h"
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

	void* ShaderProcessor::Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		ShaderProcessorInclude include;
		ComPtr<ID3DBlob> shader;
		ComPtr<ID3DBlob> error;
		HRESULT hr = D3DCompileFromFile(StringConverter::StringToWide(path).c_str(), nullptr, &include, entryPoint, model, flags, 0, shader.GetAddressOf(), error.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader: " + std::string(path.begin(), path.end()));
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
