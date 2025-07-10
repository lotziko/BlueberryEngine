#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Window.h"

namespace Blueberry
{
	struct WindowProperties;

	class WindowsWindow : public Window
	{
	public:
		WindowsWindow(const WindowProperties& properties);
		virtual ~WindowsWindow();

		virtual bool IsActive() final;
		virtual bool ProcessMessages() final;

		virtual void* GetHandle() final;

		virtual int GetWidth() const final;
		virtual int GetHeight() const final;

		LRESULT WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	private:
		void RegisterWindowClass();

	private:
		HWND m_Handle = NULL;
		HINSTANCE m_HInstance = NULL;
		String m_WindowTitle = "";
		WString m_WindowTitleWide = L""; //Wide string representation of window title
		String m_WindowClass = "";
		WString m_WindowClassWide = L""; //Wide string representation of window class name
		int m_Width = 0;
		int m_Height = 0;
	};
}