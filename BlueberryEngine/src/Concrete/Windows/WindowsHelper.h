#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class WindowsHelper
	{
	public:
		static String GetStringLastError();
		static String GetErrorMessage(HRESULT result);
		static String GetErrorMessage(HRESULT result, String message);
	};
}