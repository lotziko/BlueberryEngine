#include "Concrete\Windows\DxcHelper.h"

#include <filesystem>
#include <fstream>

namespace Blueberry
{
	ComPtr<IDxcCompiler> DxcHelper::s_Compiler = {};
	ComPtr<IDxcLibrary> DxcHelper::s_Library = {};

	IDxcCompiler* DxcHelper::GetCompiler()
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