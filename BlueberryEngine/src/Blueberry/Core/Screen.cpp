#include "Blueberry\Core\Screen.h"

namespace Blueberry
{
	uint32_t Screen::s_Width = 0;
	uint32_t Screen::s_Height = 0;
	float Screen::s_Scale = 1.0f;
	Rectangle Screen::s_GameViewport = {};
	bool Screen::s_AllowCursorLock = false;

	uint32_t Screen::GetWidth()
	{
		return s_Width;
	}

	uint32_t Screen::GetHeight()
	{
		return s_Height;
	}

	float Screen::GetScale()
	{
		return s_Scale;
	}

	Rectangle Screen::GetGameViewport()
	{
		return s_GameViewport;
	}

	void Screen::SetGameViewport(Rectangle viewport)
	{
		s_GameViewport = viewport;
	}

	bool Screen::IsAllowCursorLock()
	{
		return s_AllowCursorLock;
	}

	void Screen::SetAllowCursorLock(bool allow)
	{
		s_AllowCursorLock = allow;
	}
}
