#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API GraphicsSettings
	{
	public:
		static bool GetScreenReflections();
		static void SetScreenReflections(bool enabled);

	private:
		static bool s_ScreenReflections;
	};
}