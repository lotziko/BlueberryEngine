#include "Blueberry\Core\Window.h"

#include "..\Core\LayerStack.h"
#include "Blueberry\Events\Event.h"

#include "Blueberry\Core\Screen.h"
#include "..\..\Concrete\Windows\WindowsWindow.h"

namespace Blueberry
{
	Window* Window::Create(const WindowProperties& properties)
	{
		return new WindowsWindow(properties);
	}

	void Window::SetScreenSize(const uint32_t& width, const uint32_t& height, const float& scale)
	{
		Screen::s_Width = width;
		Screen::s_Height = height;
		Screen::s_Scale = scale;
	}
}