#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"

#include <dxc\dxcapi.h>

namespace Blueberry
{
	class DxcHelper
	{
	public:
		static IDxcCompiler* GetCompiler();
		static IDxcLibrary* GetLibrary();
		static HRESULT Reflect(IDxcBlob* pContainer, REFIID iid, void** ppvObject);

	private:
		static ComPtr<IDxcCompiler> s_Compiler;
		static ComPtr<IDxcLibrary> s_Library;
	};
}