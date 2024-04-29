#include "bbpch.h"
#include "HLSLShaderProcessor.h"

#include "Blueberry\Tools\StringConverter.h"
#include "Blueberry\Tools\StringHelper.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	HRESULT HLSLShaderProcessorInclude::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
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

	HRESULT HLSLShaderProcessorInclude::Close(LPCVOID pData)
	{
		std::free(const_cast<void*>(pData));
		return S_OK;
	}

	HLSLShaderProcessor::~HLSLShaderProcessor()
	{
		m_Blob.Reset();
	}

	void HLSLShaderProcessor::Compile(const std::string& shaderCode, const ShaderType& type)
	{
		switch (type)
		{
		case ShaderType::Vertex:
			Compile(shaderCode, "Vertex", "vs_5_0");
			break;
		case ShaderType::Fragment:
			Compile(shaderCode, "Fragment", "ps_5_0");
			break;
		case ShaderType::Compute:
			Compile(shaderCode, "Main", "cs_5_0");
			break;
		}
	}

	void HLSLShaderProcessor::LoadBlob(const std::string& path)
	{
		HRESULT hr = D3DReadFileToBlob(StringConverter::StringToWide(path).c_str(), m_Blob.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to load shader: " + std::string(path.begin(), path.end()));
			return;
		}
	}

	void HLSLShaderProcessor::SaveBlob(const std::string& path)
	{
		if (m_Blob->GetBufferSize() > 0)
		{
			D3DWriteBlobToFile(m_Blob.Get(), StringConverter::StringToWide(path).c_str(), true);
		}
	}

	void* HLSLShaderProcessor::GetShader()
	{
		return m_Blob.Get();
	}

	bool HLSLShaderProcessor::Compile(const std::string& shaderCode, const char* entryPoint, const char* model)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		HLSLShaderProcessorInclude include = {};
		ComPtr<ID3DBlob> error;

		HRESULT hr = D3DCompile2(shaderCode.data(), shaderCode.size(), nullptr, nullptr, &include, entryPoint, model, flags, 0, 0, nullptr, 0, m_Blob.GetAddressOf(), error.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader.");
			BB_ERROR((char*)error->GetBufferPointer());
			error->Release();
			return false;
		}
		return true;
	}
}
