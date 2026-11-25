#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API GraphicsSettings
	{
	public:
		static bool GetScreenReflections();
		static void SetScreenReflections(bool enabled);

		static float GetDebugValue0();
		static void SetDebugValue0(float value);

		static float GetDebugValue1();
		static void SetDebugValue1(float value);

	private:
		static bool s_ScreenReflections;
		static float s_DebugValue0;
		static float s_DebugValue1;
	};
}