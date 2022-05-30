#include "bbpch.h"
#include "WindowsHelper.h"
#include <comdef.h>

std::string WindowsHelper::GetStringLastError()
{
	DWORD dwErrorCode = GetLastError();
	if (dwErrorCode == 0) {
		return std::string(); //No error message has been recorded
	}
	else {
		return std::system_category().message(dwErrorCode);
	}
}

std::string WindowsHelper::GetErrorMessage(HRESULT result)
{
	_com_error error(result);
	std::wstring str = error.ErrorMessage();
	return std::string(str.begin(), str.end());
}

std::string WindowsHelper::GetErrorMessage(HRESULT result, std::string message)
{
	return WindowsHelper::GetErrorMessage(result) + " " + message;
}