#pragma once

namespace Blueberry
{
	class Screen
	{
	public:
		static const UINT& GetWidth();
		static const UINT& GetHeight();

	private:
		static UINT s_Width;
		static UINT s_Height;

		friend class Window;
	};
}