#include "bbpch.h"
#include "Window.h"

#include "Concrete\Windows\WindowsWindow.h"

Scope<Window> Window::Create(const WindowProperties& properties)
{
	return CreateScope<WindowsWindow>(properties);
}
