#include "Concrete\Windows\DxcHelper.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	ComPtr<IDxcCompiler3> DxcHelper::s_Compiler = {};
	ComPtr<IDxcLibrary> DxcHelper::s_Library = {};
	ComPtr<IDxcUtils> DxcHelper::s_Utils = {};

	IDxcCompiler3* DxcHelper::GetCompiler()
	{
		if (s_Compiler == nullptr)
		{
			HRESULT hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&s_Compiler));
		}
		return s_Compiler.Get();
	}

	IDxcLibrary* DxcHelper::GetLibrary()
	{
		if (s_Library == nullptr)
		{
			HRESULT hr = DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&s_Library));
		}
		return s_Library.Get();
	}

	IDxcUtils* DxcHelper::GetUtils()
	{
		if (s_Utils == nullptr)
		{
			HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&s_Utils));
		}
		return s_Utils.Get();
	}

	HRESULT DxcHelper::Reflect(IDxcBlob* pContainer, REFIID iid, void** ppvObject)
	{
		ComPtr<IDxcContainerReflection> containerReflection;
		HRESULT hr = DxcCreateInstance(CLSID_DxcContainerReflection, IID_PPV_ARGS(&containerReflection));
		if (FAILED(hr))
		{
			return hr;
		}
		hr = containerReflection->Load(pContainer);
		if (FAILED(hr))
		{
			return hr;
		}
		UINT32 partIndex;
		hr = containerReflection->FindFirstPartKind(DXC_PART_REFLECTION_DATA, &partIndex);
		if (FAILED(hr))
		{
			return hr;
		}
		hr = containerReflection->GetPartReflection(partIndex, iid, ppvObject);
		return hr;
	}
}