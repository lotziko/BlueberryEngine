#pragma once

#include "Blueberry\Graphics\Shader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

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

		bool Compile(const String& path);
		void SaveVariants(const String& folderPath);
		bool LoadVariants(const String& folderPath);

		const ShaderData& GetShaderData();
		const VariantsData& GetVariantsData();

	private:
		bool Compile(const String& shaderCode, const char* entryPoint, const char* model, D3D_SHADER_MACRO* keywords, ComPtr<ID3DBlob>& blob);
	
	private:
		ShaderData m_ShaderData;
		VariantsData m_VariantsData;
		List<ComPtr<ID3DBlob>> m_Blobs;
	};
}