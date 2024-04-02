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
		static void Process(const std::string& path, std::string& shaderCode, std::map<std::string, std::string>& options);
		static void* Compile(const std::string& shaderData, const char* entryPoint, const char* model, const std::string& blobPath);
		static void* Load(const std::string& path);
	};
}