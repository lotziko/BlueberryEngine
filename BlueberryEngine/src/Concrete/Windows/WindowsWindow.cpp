#include "bbpch.h"
#include "WindowsWindow.h"

#include "Blueberry\Events\WindowEvent.h"
#include "Blueberry\Events\KeyEvent.h"

#include "Blueberry\Tools\StringConverter.h"
#include "imgui\imgui.h"

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace Blueberry
{
	WindowsWindow::WindowsWindow(const WindowProperties& properties)
	{
		std::string windowTitle = properties.Title;
		std::string windowClass = "WindowClass";

		m_HInstance = *(static_cast<HINSTANCE*>(properties.Data));
		m_WindowTitle = windowTitle;
		m_WindowTitleWide = StringConverter::StringToWide(windowTitle);
		m_WindowClass = windowClass;
		m_WindowClassWide = StringConverter::StringToWide(windowClass);
		m_Width = properties.Width;
		m_Height = properties.Height;

		this->RegisterWindowClass();

		int centerX = (GetSystemMetrics(SM_CXSCREEN) - properties.Width) / 2;
		int centerY = (GetSystemMetrics(SM_CYSCREEN) - properties.Height) / 2;

		RECT wr;
		wr.left = centerX;
		wr.top = centerY;
		wr.right = wr.left + properties.Width;
		wr.bottom = wr.top + properties.Height;

		AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

		m_Handle = CreateWindowEx(0, //Extended Windows style - we are using the default. For other options, see: https://msdn.microsoft.com/en-us/library/windows/desktop/ff700543(v=vs.85).aspx
			m_WindowClassWide.c_str(), //Window class name
			m_WindowTitleWide.c_str(), //Window Title
			WS_OVERLAPPEDWINDOW, //Windows style - See: https://msdn.microsoft.com/en-us/library/windows/desktop/ms632600(v=vs.85).aspx
			wr.left, //Window X Position
			wr.top, //Window Y Position
			wr.right - wr.left, //Window Width
			wr.bottom - wr.top, //Window Height
			NULL, //Handle to parent of this window. Since this is the first window, it has no parent window.
			NULL, //Handle to menu or child window identifier. Can be set to NULL and use menu in WindowClassEx if a menu is desired to be used.
			m_HInstance, //Handle to the instance of module to be used with this window
			this); //Param to create window

		if (m_Handle == NULL)
		{
			BB_ERROR(WindowsHelper::GetStringLastError() + "CreateWindowEX Failed for window: " + m_WindowTitle);
		}

		ShowWindow(m_Handle, SW_SHOW);
		SetForegroundWindow(m_Handle);
		SetFocus(m_Handle);
	}

	WindowsWindow::~WindowsWindow()
	{
		if (m_Handle != NULL)
		{
			UnregisterClass(m_WindowClassWide.c_str(), m_HInstance);
			DestroyWindow(m_Handle);
		}
	}

	bool WindowsWindow::ProcessMessages()
	{
		MSG msg;
		ZeroMemory(&msg, sizeof(MSG));

		while (PeekMessage(&msg, //Where to store message (if one exists) See: https://msdn.microsoft.com/en-us/library/windows/desktop/ms644943(v=vs.85).aspx
			m_Handle, //Handle to window we are checking messages for
			0,    //Minimum Filter Msg Value - We are not filtering for specific messages, but the min/max could be used to filter only mouse messages for example.
			0,    //Maximum Filter Msg Value
			PM_REMOVE))//Remove message after capturing it via PeekMessage. For more argument options, see: https://msdn.microsoft.com/en-us/library/windows/desktop/ms644943(v=vs.85).aspx
		{
			TranslateMessage(&msg); //Translate message from virtual key messages into character messages so we can dispatch the message. See: https://msdn.microsoft.com/en-us/library/windows/desktop/ms644955(v=vs.85).aspx
			DispatchMessage(&msg); //Dispatch message to our Window Proc for this window. See: https://msdn.microsoft.com/en-us/library/windows/desktop/ms644934(v=vs.85).aspx
		}

		// Check if the window was closed
		if (msg.message == WM_NULL)
		{
			if (!IsWindow(m_Handle))
			{
				m_Handle = NULL; //Message processing loop takes care of destroying this window
				UnregisterClass(m_WindowClassWide.c_str(), m_HInstance);
				return false;
			}
		}

		return true;
	}

	void* WindowsWindow::GetHandle()
	{
		return &m_Handle;
	}

	int WindowsWindow::GetWidth() const
	{
		return m_Width;
	}

	int WindowsWindow::GetHeight() const
	{
		return m_Height;
	}

	LRESULT CALLBACK HandleMsgRedirect(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		switch (uMsg)
		{
		case WM_CLOSE:
		{
			DestroyWindow(hwnd);
			return 0;
			break;
		}
		default:
		{
			WindowsWindow* pWindow = reinterpret_cast<WindowsWindow*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
			return pWindow->WindowProc(hwnd, uMsg, wParam, lParam);
			break;
		}
		}
	}

	LRESULT CALLBACK HandleMessageSetup(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		switch (uMsg)
		{
		case WM_NCCREATE:
		{
			const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
			WindowsWindow* pWindow = reinterpret_cast<WindowsWindow*>(pCreate->lpCreateParams);
			if (pWindow == nullptr)
			{
				BB_ERROR("Critical Error: Pointer to window container is null during WM_NCCREATE.");
				exit(-1);
			}
			SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWindow));
			SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(HandleMsgRedirect));
			return pWindow->WindowProc(hwnd, uMsg, wParam, lParam);
			break;
		}
		default:
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
			break;
		}
	}

	LRESULT WindowsWindow::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		if (ImGui_ImplWin32_WndProcHandler(hwnd, uMsg, wParam, lParam))
			return true;

		switch (uMsg)
		{
		case WM_KEYDOWN:
		{
			unsigned char key = static_cast<unsigned char>(wParam);
			KeyPressedEvent event(key);
			EventDispatcher::Invoke(event);
			return 0;
		}
		case WM_KEYUP:
		{
			unsigned char key = static_cast<unsigned char>(wParam);
			KeyReleasedEvent event(key);
			EventDispatcher::Invoke(event);
			return 0;
		}
		case WM_CHAR:
		{
			unsigned char ch = static_cast<unsigned char>(wParam);
			KeyTypedEvent event(ch);
			EventDispatcher::Invoke(event);
			return 0;
		}
		case WM_SIZE:
		{
			UINT width = LOWORD(lParam);
			UINT height = HIWORD(lParam);
			if (width != m_Width || height != m_Height)
			{
				m_Width = width;
				m_Height = height;

				WindowResizeEvent event(width, height);
				EventDispatcher::Invoke(event);
			}
			return 0;
		}

		default:
			break;
		}
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}

	void WindowsWindow::RegisterWindowClass()
	{
		WNDCLASSEX wc; //Our Window Class (This has to be filled before our window can be created) See: https://msdn.microsoft.com/en-us/library/windows/desktop/ms633577(v=vs.85).aspx
		wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; //Flags [Redraw on width/height change from resize/movement] See: https://msdn.microsoft.com/en-us/library/windows/desktop/ff729176(v=vs.85).aspx
		wc.lpfnWndProc = HandleMessageSetup; //Pointer to Window Proc function for handling messages from this window
		wc.cbClsExtra = 0; //# of extra bytes to allocate following the window-class structure. We are not currently using this.
		wc.cbWndExtra = 0; //# of extra bytes to allocate following the window instance. We are not currently using this.
		wc.hInstance = m_HInstance; //Handle to the instance that contains the Window Procedure
		wc.hIcon = NULL;   //Handle to the class icon. Must be a handle to an icon resource. We are not currently assigning an icon, so this is null.
		wc.hIconSm = NULL; //Handle to small icon for this class. We are not currently assigning an icon, so this is null.
		wc.hCursor = LoadCursor(NULL, IDC_ARROW); //Default Cursor - If we leave this null, we have to explicitly set the cursor's shape each time it enters the window.
		wc.hbrBackground = NULL; //Handle to the class background brush for the window's background color - we will leave this blank for now and later set this to black. For stock brushes, see: https://msdn.microsoft.com/en-us/library/windows/desktop/dd144925(v=vs.85).aspx
		wc.lpszMenuName = NULL; //Pointer to a null terminated character string for the menu. We are not using a menu yet, so this will be NULL.
		wc.lpszClassName = m_WindowClassWide.c_str(); //Pointer to null terminated string of our class name for this window.
		wc.cbSize = sizeof(WNDCLASSEX); //Need to fill in the size of our struct for cbSize
		RegisterClassEx(&wc); // Register the class so that it is usable.
	}
}