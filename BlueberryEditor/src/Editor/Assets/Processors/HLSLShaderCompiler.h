#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"

#include <d3dcompiler.h>
#include <dxc\dxcapi.h>

namespace Blueberry
{
	enum class HLSLShaderCompilerProfile
	{
		None,
		Vertex,
		Geometry,
		Fragment,
		Compute
	};

	class HLSLShaderCompiler
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual void SetKeywords(const List<String>& keywords) = 0;
		virtual bool Compile(const String& entryPoint, HLSLShaderCompilerProfile model, uint32_t variant, ByteData& result) = 0;
	};

	class HLSLShaderProcessorInclude : public ID3DInclude
	{
		HRESULT Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;
		HRESULT Close(LPCVOID pData) override;
	};

	class HLSLShaderCompilerFXC : public HLSLShaderCompiler
	{
	public:
		HLSLShaderCompilerFXC(const String& code);

		virtual void SetKeywords(const List<String>& keywords) override;
		virtual bool Compile(const String& entryPoint, HLSLShaderCompilerProfile profile, uint32_t variant, ByteData& result) override;

	private:
		String m_Code;
		List<String> m_Keywords;
		List<D3D_SHADER_MACRO> m_Defines;
	};

	class HLSLShaderCompilerIncludeHandler : public IDxcIncludeHandler
	{
		virtual HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) override;

		virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
		virtual ULONG STDMETHODCALLTYPE AddRef() override;
		virtual ULONG STDMETHODCALLTYPE Release() override;
	};

	class HLSLShaderCompilerDXC : public HLSLShaderCompiler
	{
	public:
		HLSLShaderCompilerDXC(const String& code);

		virtual void SetKeywords(const List<String>& keywords) override;
		virtual bool Compile(const String& entryPoint, HLSLShaderCompilerProfile profile, uint32_t variant, ByteData& result) override;

	private:
		ComPtr<IDxcBlobEncoding> m_CodeBlob;
		List<WString> m_Keywords;
		List<DxcDefine> m_Defines;
	};
}