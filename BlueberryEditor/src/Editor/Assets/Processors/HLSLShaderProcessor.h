#pragma once

#include "Blueberry\Graphics\Shader.h"

namespace Blueberry
{
	class HLSLShaderProcessorInclude : public ID3DInclude
	{
		HRESULT Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;
		HRESULT Close(LPCVOID pData) override;
	};

	class HLSLShaderProcessor
	{
	public:
		HLSLShaderProcessor() = default;
		~HLSLShaderProcessor();

		bool Compile(const std::string& path);
		void SaveVariants(const std::string& folderPath);
		bool LoadVariants(const std::string& folderPath);

		const ShaderData& GetShaderData();
		const VariantsData& GetVariantsData();

	private:
		bool Compile(const std::string& shaderCode, const char* entryPoint, const char* model, D3D_SHADER_MACRO* keywords, ComPtr<ID3DBlob>& blob);
	
	private:
		ShaderData m_ShaderData;
		VariantsData m_VariantsData;
		std::vector<ComPtr<ID3DBlob>> m_Blobs;
	};
}