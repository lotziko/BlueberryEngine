#include "bbpch.h"
#include "ShaderProcessor.h"
#include "Blueberry\Tools\StringConverter.h"

namespace Blueberry
{
	void* ShaderProcessor::Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath)
	{
		UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

		ComPtr<ID3D10Blob> shader;
		HRESULT hr = D3DCompileFromFile(StringConverter::StringToWide(path).c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, model, flags, 0, shader.GetAddressOf(), nullptr);
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to compile shader: " + std::string(path.begin(), path.end());
			BB_ERROR(errorMsg);
			return nullptr;
		}
		D3DWriteBlobToFile(shader.Get(), StringConverter::StringToWide(blobPath).c_str(), true);
		return shader.Detach();
	}

	void* ShaderProcessor::Load(const std::string& path)
	{
		ComPtr<ID3D10Blob> shader;
		HRESULT hr = D3DReadFileToBlob(StringConverter::StringToWide(path).c_str(), shader.GetAddressOf());
		if (FAILED(hr))
		{
			std::string errorMsg = "Failed to load shader: " + std::string(path.begin(), path.end());
			BB_ERROR(errorMsg);
			return nullptr;
		}
		return shader.Detach();
	}
}
