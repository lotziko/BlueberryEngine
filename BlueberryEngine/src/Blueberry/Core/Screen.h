#pragma once

namespace Blueberry
{
	class Screen
	{
	public:
		static const uint32_t& GetWidth();
		static const uint32_t& GetHeight();

	private:
		static uint32_t s_Width;
		static uint32_t s_Height;

		friend class Window;
	};
}