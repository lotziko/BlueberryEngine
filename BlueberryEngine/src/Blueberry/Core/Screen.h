#pragma once

namespace Blueberry
{
	class Screen
	{
	public:
		static const uint32_t& GetWidth();
		static const uint32_t& GetHeight();
		static const float& GetScale();

	private:
		static uint32_t s_Width;
		static uint32_t s_Height;
		static float s_Scale;

		friend class Window;
	};
}