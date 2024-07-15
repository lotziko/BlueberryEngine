#pragma once

#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class WindowResizeEventArgs
	{
	public:
		WindowResizeEventArgs(UINT width, UINT height) : m_Width(width), m_Height(height)
		{
		}

		UINT GetWidth() const;
		UINT GetHeight() const;

	private:
		UINT m_Width;
		UINT m_Height;
	};

	using WindowResizeEvent = Event<WindowResizeEventArgs>;

	class WindowEvents
	{
	public:
		static WindowResizeEvent& GetWindowResized();

	private:
		static WindowResizeEvent s_WindowResized;
	};
}