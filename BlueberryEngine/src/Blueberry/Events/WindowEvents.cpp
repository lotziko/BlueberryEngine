#include "bbpch.h"
#include "WindowEvents.h"

namespace Blueberry
{
	WindowResizeEvent WindowEvents::s_WindowResized = {};

	UINT WindowResizeEventArgs::GetWidth() const
	{
		return m_Width;
	}

	UINT WindowResizeEventArgs::GetHeight() const
	{
		return m_Height;
	}

	WindowResizeEvent& WindowEvents::GetWindowResized()
	{
		return s_WindowResized;
	}
}