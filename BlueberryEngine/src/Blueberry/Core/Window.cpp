#include "bbpch.h"
#include "Window.h"

#include "Concrete\Windows\WindowsWindow.h"

namespace Blueberry
{
	Window* Window::Create(const WindowProperties& properties)
	{
		return new WindowsWindow(properties);
	}
}