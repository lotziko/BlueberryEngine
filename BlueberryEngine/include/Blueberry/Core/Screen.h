#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API Screen
	{
	public:
		static uint32_t GetWidth();
		static uint32_t GetHeight();
		static float GetScale();

		static Rectangle GetGameViewport();
		static void SetGameViewport(Rectangle viewport);

		static bool IsAllowCursorLock();
		static void SetAllowCursorLock(bool allow);

	private:
		static uint32_t s_Width;
		static uint32_t s_Height;
		static Rectangle s_GameViewport;
		static bool s_AllowCursorLock;
		static float s_Scale;

		friend class Window;
	};
}