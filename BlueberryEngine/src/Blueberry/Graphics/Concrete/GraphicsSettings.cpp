#include "Blueberry\Graphics\Concrete\GraphicsSettings.h"

namespace Blueberry
{
	bool GraphicsSettings::s_ScreenReflections = true;

	bool GraphicsSettings::GetScreenReflections()
	{
		return s_ScreenReflections;
	}

	void GraphicsSettings::SetScreenReflections(bool enabled)
	{
		s_ScreenReflections = enabled;
	}
}
