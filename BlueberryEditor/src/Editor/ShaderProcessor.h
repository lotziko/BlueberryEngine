#pragma once

namespace Blueberry
{
	class ShaderProcessorInclude : public ID3DInclude
	{
		HRESULT Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;
		HRESULT Close(LPCVOID pData) override;
	};

	class ShaderProcessor
	{
	public:
		static void* Compile(const std::string& path, const char* entryPoint, const char* model, const std::string& blobPath);
		static void* Load(const std::string& path);
	};
}