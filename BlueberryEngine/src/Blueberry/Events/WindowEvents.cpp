#include "Blueberry\Events\WindowEvents.h"

namespace Blueberry
{
	WindowResizeEvent WindowEvents::s_WindowResized = {};
	WindowFocusEvent WindowEvents::s_WindowFocused = {};
	WindowUnfocusEvent WindowEvents::s_WindowUnfocused = {};
	WindowDropFilesEvent WindowEvents::s_WindowDroppedFiles = {};
	WindowClosingEvent WindowEvents::s_WindowClosing = {};

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

	WindowUnfocusEvent& WindowEvents::GetWindowUnfocused()
	{
		return s_WindowUnfocused;
	}

	WindowDropFilesEvent& WindowEvents::GetWindowDroppedFiles()
	{
		return s_WindowDroppedFiles;
	}

	WindowClosingEvent& WindowEvents::GetWindowClosing()
	{
		return s_WindowClosing;
	}

	void WindowDropFilesEventArgs::AddFile(const WString& path)
	{
		m_Pathes.push_back(path);
	}

	const List<WString>& WindowDropFilesEventArgs::GetFiles() const
	{
		return m_Pathes;
	}

	const bool& WindowClosingEventArgs::IsCanceled()
	{
		return m_IsCanceled;
	}

	void WindowClosingEventArgs::Cancel()
	{
		m_IsCanceled = true;
	}
}