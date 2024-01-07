#include "bbpch.h"
#include "ShaderProcessor.h"
#include "Blueberry\Tools\StringConverter.h"

namespace Blueberry
{
	void* ShaderProcessor::Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		ComPtr<ID3DBlob> shader;
		ComPtr<ID3DBlob> error;
		HRESULT hr = D3DCompileFromFile(StringConverter::StringToWide(path).c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, model, flags, 0, shader.GetAddressOf(), error.GetAddressOf());
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
