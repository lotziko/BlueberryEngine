#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class WindowResizeEventArgs
	{
	public:
		WindowResizeEventArgs(uint32_t width, uint32_t height) : m_Width(width), m_Height(height)
		{
		}

		uint32_t GetWidth() const;
		uint32_t GetHeight() const;

	private:
		uint32_t m_Width;
		uint32_t m_Height;
	};

	using WindowResizeEvent = Event<WindowResizeEventArgs>;
	using WindowFocusEvent = Event<>;
	using WindowUnfocusEvent = Event<>;

	class WindowEvents
	{
	public:
		static WindowResizeEvent& GetWindowResized();
		static WindowFocusEvent& GetWindowFocused();
		static WindowUnfocusEvent& GetWindowUnfocused();

	private:
		static WindowResizeEvent s_WindowResized;
		static WindowFocusEvent s_WindowFocused;
		static WindowUnfocusEvent s_WindowUnfocused;
	};
}