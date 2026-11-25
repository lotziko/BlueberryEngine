#include "Blueberry\Graphics\Concrete\GraphicsSettings.h"

namespace Blueberry
{
	bool GraphicsSettings::s_ScreenReflections = false;
	float GraphicsSettings::s_DebugValue0 = 0;
	float GraphicsSettings::s_DebugValue1 = 0;

	bool GraphicsSettings::GetScreenReflections()
	{
		return s_ScreenReflections;
	}

	void GraphicsSettings::SetScreenReflections(bool enabled)
	{
		s_ScreenReflections = enabled;
	}

	float GraphicsSettings::GetDebugValue0()
	{
		return s_DebugValue0;
	}

	void GraphicsSettings::SetDebugValue0(float value)
	{
		s_DebugValue0 = value;
	}

	float GraphicsSettings::GetDebugValue1()
	{
		return s_DebugValue1;
	}

	void GraphicsSettings::SetDebugValue1(float value)
	{
		s_DebugValue1 = value;
	}
}
