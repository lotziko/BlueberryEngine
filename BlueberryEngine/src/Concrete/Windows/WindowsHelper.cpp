#include "WindowsHelper.h"
#include <comdef.h>

namespace Blueberry
{
	String WindowsHelper::GetStringLastError()
	{
		DWORD dwErrorCode = GetLastError();
		if (dwErrorCode == 0) {
			return String(); //No error message has been recorded
		}
		else {
			return String(std::system_category().message(dwErrorCode));
		}
	}

	String WindowsHelper::GetErrorMessage(HRESULT result)
	{
		_com_error error(result);
		std::wstring str = error.ErrorMessage();
		return String(str.begin(), str.end());
	}

	String WindowsHelper::GetErrorMessage(HRESULT result, String message)
	{
		return WindowsHelper::GetErrorMessage(result) + " " + message;
	}
}