#include "HLSLShaderCompiler.h"

#include "Blueberry\Tools\StringHelper.h"
#include "Concrete\Windows\DxcHelper.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	HRESULT HLSLShaderProcessorInclude::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes)
	{
		String filePath("assets/shaders/" + String(pFileName));
		if (!std::filesystem::exists(filePath))
		{
			return E_FAIL;
		}

		// Based on https://github.com/holy-shit/clion-directx-example/blob/master/main.cpp
		std::streampos dataSize;
		char* buffer;

		std::ifstream infile;
		infile.open(filePath.data(), std::ios::binary);
		infile.seekg(0, std::ios::end);
		dataSize = infile.tellg();
		infile.seekg(0, std::ios::beg);
		buffer = static_cast<char*>(malloc(dataSize));
		infile.read(buffer, dataSize);
		infile.close();

		*pBytes = static_cast<UINT>(dataSize);
		*ppData = buffer;

		return S_OK;
	}

	HRESULT HLSLShaderProcessorInclude::Close(LPCVOID pData)
	{
		free(const_cast<void*>(pData));
		return S_OK;
	}

	HLSLShaderCompilerFXC::HLSLShaderCompilerFXC(const String& code)
	{
		m_Code = code;
	}

	void HLSLShaderCompilerFXC::SetKeywords(const List<String>& keywords)
	{
		m_Keywords.reserve(keywords.size());
		m_Keywords.clear();
		m_Defines.clear();
		for (auto& keyword : keywords)
		{
			m_Keywords.push_back(keyword);
			D3D_SHADER_MACRO define = {};
			define.Name = m_Keywords.back().c_str();
			m_Defines.push_back(define);
		}
		m_Defines.push_back({});
	}

	bool HLSLShaderCompilerFXC::Compile(const String& entryPoint, HLSLShaderCompilerProfile profile, uint32_t variant, ByteData& result)
	{
		const char* targetProfile = nullptr;
		switch (profile)
		{
		case HLSLShaderCompilerProfile::Vertex:
			targetProfile = "vs_5_0";
			break;
		case HLSLShaderCompilerProfile::Geometry:
			targetProfile = "gs_5_0";
			break;
		case HLSLShaderCompilerProfile::Fragment:
			targetProfile = "ps_5_0";
			break;
		case HLSLShaderCompilerProfile::Compute:
			targetProfile = "cs_5_0";
			break;
		default:
			return false;
		}

		if (m_Defines.size() > 0)
		{
			for (size_t i = 0; i < m_Defines.size() - 1; ++i)
			{
				m_Defines[i].Definition = (1ull << i) & variant ? "1" : "0";
			}
		}

		uint32_t flags = D3DCOMPILE_ENABLE_STRICTNESS;

		HLSLShaderProcessorInclude include = {};
		ComPtr<ID3DBlob> temporaryBlob;
		ComPtr<ID3DBlob> error;

		HRESULT hr = D3DCompile2(m_Code.data(), m_Code.size(), nullptr, m_Defines.data(), &include, entryPoint.c_str(), targetProfile, flags, 0, 0, nullptr, 0, temporaryBlob.GetAddressOf(), error.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader.");
			BB_ERROR(static_cast<char*>(error->GetBufferPointer()));
			error->Release();
			return false;
		}

		ComPtr<ID3DBlob> resultBlob;
		hr = D3DStripShader(temporaryBlob->GetBufferPointer(), temporaryBlob->GetBufferSize(), D3DCOMPILER_STRIP_DEBUG_INFO | D3DCOMPILER_STRIP_TEST_BLOBS, resultBlob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to strip shader.");
			return false;
		}

		result.resize(resultBlob->GetBufferSize());
		memcpy(result.data(), resultBlob->GetBufferPointer(), result.size());

		return true;
	}

	HRESULT STDMETHODCALLTYPE HLSLShaderCompilerIncludeHandler::LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource)
	{
		WString filePath(L"assets/shaders/" + WString(pFilename));
		if (!std::filesystem::exists(filePath))
		{
			return E_FAIL;
		}

		std::streampos dataSize;
		char* buffer;

		std::ifstream infile;
		infile.open(filePath.data(), std::ios::binary);
		infile.seekg(0, std::ios::end);
		dataSize = infile.tellg();
		infile.seekg(0, std::ios::beg);
		buffer = static_cast<char*>(malloc(dataSize));
		infile.read(buffer, dataSize);
		infile.close();

		if (buffer != nullptr)
		{
			ComPtr<IDxcBlobEncoding> blob;
			DxcHelper::GetLibrary()->CreateBlobWithEncodingOnHeapCopy(buffer, static_cast<UINT32>(dataSize), CP_UTF8, blob.GetAddressOf());
			*ppIncludeSource = blob.Detach();
			return S_OK;
		}
		return E_FAIL;
	}

	HRESULT STDMETHODCALLTYPE HLSLShaderCompilerIncludeHandler::QueryInterface(REFIID riid, void** ppvObject)
	{
		return E_NOINTERFACE;
	}

	ULONG STDMETHODCALLTYPE HLSLShaderCompilerIncludeHandler::AddRef()
	{
		return 1;
	}

	ULONG STDMETHODCALLTYPE HLSLShaderCompilerIncludeHandler::Release()
	{
		return 1;
	}

	HLSLShaderCompilerDXC::HLSLShaderCompilerDXC(const String& code)
	{
		DxcHelper::GetLibrary()->CreateBlobWithEncodingOnHeapCopy(code.c_str(), static_cast<UINT32>(code.size()), CP_UTF8, m_CodeBlob.GetAddressOf());
	}

	void HLSLShaderCompilerDXC::SetKeywords(const List<String>& keywords)
	{
		m_Keywords.reserve(keywords.size());
		m_Keywords.clear();
		m_Defines.clear();
		for (auto& keyword : keywords)
		{
			m_Keywords.push_back(StringHelper::StringToWide(keyword));
			DxcDefine define = {};
			define.Name = m_Keywords.back().c_str();
			m_Defines.push_back(define);
		}
	}

	bool HLSLShaderCompilerDXC::Compile(const String& entryPoint, HLSLShaderCompilerProfile profile, uint32_t variant, ByteData& result)
	{
		const wchar_t* targetProfile = nullptr;
		switch (profile)
		{
		case HLSLShaderCompilerProfile::Vertex:
			targetProfile = L"vs_6_0";
			break;
		case HLSLShaderCompilerProfile::Geometry:
			targetProfile = L"gs_6_0";
			break;
		case HLSLShaderCompilerProfile::Fragment:
			targetProfile = L"ps_6_0";
			break;
		case HLSLShaderCompilerProfile::Compute:
			targetProfile = L"cs_6_0";
			break;
		default:
			return false;
		}

		for (size_t i = 0; i < m_Defines.size(); ++i)
		{
			m_Defines[i].Value = (1ull << i) & variant ? L"1" : L"0";
		}

		HLSLShaderCompilerIncludeHandler includeHandler = {};
		ComPtr<IDxcOperationResult> operationResult;
		HRESULT hr = DxcHelper::GetCompiler()->Compile(m_CodeBlob.Get(), nullptr, StringHelper::StringToWide(entryPoint).c_str(), targetProfile, nullptr, 0, m_Defines.data(), static_cast<UINT32>(m_Defines.size()), &includeHandler, operationResult.GetAddressOf());
		operationResult->GetStatus(&hr);

		if (FAILED(hr))
		{
			BB_ERROR("Failed to compile shader.");
			ComPtr<IDxcBlobEncoding> error;
			operationResult->GetErrorBuffer(error.GetAddressOf());
			BB_ERROR(static_cast<char*>(error->GetBufferPointer()));
			return false;
		}

		ComPtr<IDxcBlob> resultBlob;
		hr = operationResult->GetResult(resultBlob.GetAddressOf());

		if (FAILED(hr))
		{
			BB_ERROR("Failed to get shader.");
			return false;
		}

		result.resize(resultBlob->GetBufferSize());
		memcpy(result.data(), resultBlob->GetBufferPointer(), result.size());

		return true;
	}
}