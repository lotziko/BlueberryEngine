#pragma once

namespace Blueberry
{
	class HLSLShaderProcessorInclude : public ID3DInclude
	{
		HRESULT Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;
		HRESULT Close(LPCVOID pData) override;
	};

	enum class ShaderType
	{
		Vertex,
		Fragment,
		Compute
	};

	class HLSLShaderProcessor
	{
	public:
		HLSLShaderProcessor() = default;
		~HLSLShaderProcessor();

		void Compile(const std::string& shaderCode, const ShaderType& type);
		void LoadBlob(const std::string& path);
		void SaveBlob(const std::string& path);
		void* GetShader();

	private:
		bool Compile(const std::string& shaderCode, const char* entryPoint, const char* model);
	
	private:
		ComPtr<ID3DBlob> m_Blob;
	};
}