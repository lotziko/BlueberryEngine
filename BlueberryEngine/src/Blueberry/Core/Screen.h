#pragma once

#include "Blueberry\Math\Math.h"

namespace Blueberry
{
	class Screen
	{
	public:
		static const uint32_t& GetWidth();
		static const uint32_t& GetHeight();
		static const float& GetScale();

		static const Rectangle& GetGameViewport();
		static void SetGameViewport(const Rectangle& viewport);

		static const bool& IsAllowCursorLock();
		static void SetAllowCursorLock(const bool& allow);

	private:
		static uint32_t s_Width;
		static uint32_t s_Height;
		static Rectangle s_GameViewport;
		static bool s_AllowCursorLock;
		static float s_Scale;

		friend class Window;
	};
}