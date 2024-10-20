#include "bbpch.h"
#include "Window.h"

#include "Blueberry\Core\Screen.h"
#include "Concrete\Windows\WindowsWindow.h"

namespace Blueberry
{
	Window* Window::Create(const WindowProperties& properties)
	{
		return new WindowsWindow(properties);
	}

	void Window::SetScreenSize(const UINT& width, const UINT& height)
	{
		Screen::s_Width = width;
		Screen::s_Height = height;
	}
}