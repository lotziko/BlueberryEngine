#include "bbpch.h"
#include "WindowEvents.h"

namespace Blueberry
{
	WindowResizeEvent WindowEvents::s_WindowResized = {};
	WindowFocusEvent WindowEvents::s_WindowFocused = {};

	uint32_t WindowResizeEventArgs::GetWidth() const
	{
		return m_Width;
	}

	uint32_t WindowResizeEventArgs::GetHeight() const
	{
		return m_Height;
	}

	WindowResizeEvent& WindowEvents::GetWindowResized()
	{
		return s_WindowResized;
	}

	WindowFocusEvent& WindowEvents::GetWindowFocused()
	{
		return s_WindowFocused;
	}
}