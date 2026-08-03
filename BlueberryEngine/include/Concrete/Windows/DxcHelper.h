#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\Windows\ComPtr.h"

#include <dxc\dxcapi.h>

namespace Blueberry
{
	class DxcHelper
	{
	public:
		static IDxcCompiler3* GetCompiler();
		static IDxcLibrary* GetLibrary();
		static IDxcUtils* GetUtils();
		static HRESULT Reflect(IDxcBlob* pContainer, REFIID iid, void** ppvObject);

	private:
		static ComPtr<IDxcCompiler3> s_Compiler;
		static ComPtr<IDxcLibrary> s_Library;
		static ComPtr<IDxcUtils> s_Utils;
	};
}