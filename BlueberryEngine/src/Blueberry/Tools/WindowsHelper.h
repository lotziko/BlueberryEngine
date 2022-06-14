#pragma once

namespace Blueberry
{
	class WindowsHelper
	{
	public:
		static std::string GetStringLastError();
		static std::string GetErrorMessage(HRESULT result);
		static std::string GetErrorMessage(HRESULT result, std::string message);
	};
}