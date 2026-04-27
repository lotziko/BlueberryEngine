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

	class WindowDropFilesEventArgs
	{
	public:
		const List<WString>& GetFiles() const;
		void AddFile(const WString& path);

	private:
		List<WString> m_Pathes;
	};

	class WindowClosingEventArgs
	{
	public:
		bool IsCanceled() const;
		void Cancel();
		
	private:
		bool m_IsCanceled;
	};

	using WindowResizeEvent = Event<const WindowResizeEventArgs&>;
	using WindowFocusEvent = Event<>;
	using WindowUnfocusEvent = Event<>;
	using WindowDropFilesEvent = Event<const WindowDropFilesEventArgs&>;
	using WindowClosingEvent = Event<WindowClosingEventArgs&>;

	class WindowEvents
	{
	public:
		static WindowResizeEvent& GetWindowResized();
		static WindowFocusEvent& GetWindowFocused();
		static WindowUnfocusEvent& GetWindowUnfocused();
		static WindowDropFilesEvent& GetWindowDroppedFiles();
		static WindowClosingEvent& GetWindowClosing();

	private:
		static WindowResizeEvent s_WindowResized;
		static WindowFocusEvent s_WindowFocused;
		static WindowUnfocusEvent s_WindowUnfocused;
		static WindowDropFilesEvent s_WindowDroppedFiles;
		static WindowClosingEvent s_WindowClosing;
	};
}