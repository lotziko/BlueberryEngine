#include "PlatformHelper.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Window.h"

#include <shobjidl.h> // IFileOpenDialog
#include <shlobj.h> // SHOpenFolderAndSelectItems
#include <commctrl.h>

#include <atomic>

// https://cybernaught.net/code/codeAPItaskdialog.html
#pragma comment(lib, "comctl32.lib")
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace Blueberry
{
	static DWORD s_ProgressThreadId = 0;

	#define WM_SHOWPROGRESS (WM_USER + 1)
	#define WM_HIDEPROGRESS (WM_USER + 2)

	WString PlatformHelper::OpenFileDialog()
	{
		WString result;
		IFileOpenDialog* dialog = nullptr;

		HRESULT hr = CoCreateInstance(
			CLSID_FileOpenDialog,
			nullptr,
			CLSCTX_INPROC_SERVER,
			IID_PPV_ARGS(&dialog)
		);

		if (FAILED(hr))
		{
			return L"";
		}

		DWORD options;
		dialog->GetOptions(&options);
		dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);

		hr = dialog->Show(NULL);
		if (SUCCEEDED(hr))
		{
			IShellItem* item = nullptr;
			if (SUCCEEDED(dialog->GetResult(&item)))
			{
				PWSTR path = nullptr;
				if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)))
				{
					result = path;
					CoTaskMemFree(path);
					item->Release();
					dialog->Release();
				}
				item->Release();
			}
		}

		dialog->Release();
		return result;
	}

	HRESULT CALLBACK TaskDialogCallbackProc(HWND hwnd, UINT uNotification, WPARAM wParam, LPARAM lParam, LONG_PTR dwRefData)
	{
		switch (uNotification)
		{
		case TDN_CREATED:
			SendMessage(hwnd, WM_SETICON, ICON_BIG, NULL);
			SendMessage(hwnd, WM_SETICON, ICON_SMALL, NULL);
			break;
		}
		return S_OK;
	}

	DialogResult PlatformHelper::OpenDialog(const WString& titleText, const WString& contentText, const WString& yesText, const WString& noText, const WString& cancelText)
	{
		TASKDIALOG_BUTTON buttons[] =
		{
			{ IDYES, yesText.c_str() },
			{ IDNO, noText.c_str() },
			{ IDCANCEL, cancelText.c_str() }
		};

		TASKDIALOGCONFIG config = {};
		config.cbSize = sizeof(config);
		config.hwndParent = NULL;
		config.dwFlags = TDF_ALLOW_DIALOG_CANCELLATION;
		config.dwCommonButtons = 0;
		config.pszWindowTitle = titleText.c_str();
		config.pszMainIcon = TD_WARNING_ICON;
		config.pszMainInstruction = NULL;
		config.pszContent = contentText.c_str();
		config.pButtons = buttons;
		config.cButtons = ARRAYSIZE(buttons);
		config.nDefaultButton = IDYES;
		config.pfCallback = TaskDialogCallbackProc;

		int result = 0;
		TaskDialogIndirect(&config, &result, nullptr, nullptr);

		switch (result)
		{
		case IDYES:
			return DialogResult::Yes;
		case IDNO:
			return DialogResult::No;
		default:
			return DialogResult::Cancel;
		}
	}

	void PlatformHelper::RevealInExplorer(const WString& path)
	{
		PIDLIST_ABSOLUTE pidl = nullptr;
		HRESULT hr = SHParseDisplayName(
			path.c_str(),
			nullptr,
			&pidl,
			0,
			nullptr
		);

		if (FAILED(hr))
		{
			return;
		}

		hr = SHOpenFolderAndSelectItems(
			pidl,
			0,
			nullptr,
			0
		);

		CoTaskMemFree(pidl);
	}

	void CenterWindowOnMonitor(HWND hwnd)
	{
		RECT rc{};
		GetWindowRect(hwnd, &rc);
		int winW = rc.right - rc.left;
		int winH = rc.bottom - rc.top;

		HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);

		MONITORINFO mi{};
		mi.cbSize = sizeof(mi);
		GetMonitorInfoW(mon, &mi);

		RECT work = mi.rcWork;

		int x = work.left + (work.right - work.left - winW) / 2;
		int y = work.top + (work.bottom - work.top - winH) / 2;

		SetWindowPos(
			hwnd,
			nullptr,
			x, y,
			0, 0,
			SWP_NOZORDER | SWP_NOSIZE
		);
	}

	LRESULT CALLBACK ProgressWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{
		switch (msg)
		{
		case WM_CREATE:
			CenterWindowOnMonitor(hwnd);
			break;
		case WM_CTLCOLORSTATIC:
		{
			HDC hdc = (HDC)wParam;
			SetBkMode(hdc, TRANSPARENT);
			SetTextColor(hdc, GetSysColor(COLOR_CAPTIONTEXT));
			return (LRESULT)GetSysColorBrush(COLOR_WINDOW);
		}
		case WM_SYSCOMMAND:
			if ((wParam & 0xFFF0) == SC_MOVE)
			{
				return 0;
			}
			break;
		case WM_CLOSE:
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		}

		return DefWindowProcW(hwnd, msg, wParam, lParam);
	}

	struct ProgressWindowData
	{
		HWND windowHandle;
		HWND labelHandle;
		HFONT labelFont;
	};

	ProgressWindowData CreateProgressWindow()
	{
		ProgressWindowData data;

		WNDCLASSW wc{};
		wc.lpfnWndProc = ProgressWndProc;
		wc.hInstance = GetModuleHandle(nullptr);
		wc.lpszClassName = L"ProgressWindow";
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);

		RegisterClassW(&wc);

		data.windowHandle = CreateWindowExW(
			WS_EX_DLGMODALFRAME | WS_EX_DLGMODALFRAME | WS_EX_NOACTIVATE,
			L"ProgressWindow",
			NULL,
			WS_CAPTION | WS_VISIBLE,
			CW_USEDEFAULT, CW_USEDEFAULT, 420, 160,
			NULL,
			NULL,
			wc.hInstance,
			NULL
		);

		HWND progressBarHandle = CreateWindowExW(
			0,
			PROGRESS_CLASSW,
			nullptr,
			WS_CHILD | WS_VISIBLE | PBS_MARQUEE,
			20, 50, 360, 20,
			data.windowHandle,
			nullptr,
			GetModuleHandle(nullptr),
			nullptr
		);
		SendMessage(progressBarHandle, PBM_SETMARQUEE, TRUE, 0);

		data.labelHandle = CreateWindowEx(
			0,
			L"STATIC",
			L"Test",
			WS_CHILD | WS_VISIBLE | SS_LEFT,
			20, 80, 360, 20,
			data.windowHandle,
			nullptr,
			GetModuleHandle(nullptr),
			nullptr
		);

		NONCLIENTMETRICS ncm = { sizeof(NONCLIENTMETRICS) };
		SystemParametersInfo(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0);
		data.labelFont = CreateFontIndirect(&ncm.lfCaptionFont);
		SendMessage(data.labelHandle, WM_SETFONT, (WPARAM)data.labelFont, TRUE);
		return data;
	}

	DWORD WINAPI ProgressUiThread(void*)
	{
		ProgressWindowData data = CreateProgressWindow();
		DWORD closeTime = 0;
		MSG msg;
		while (true)
		{
			while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);

				switch (msg.message)
				{
				case WM_SHOWPROGRESS:
				{
					closeTime = 0;
					auto pair = reinterpret_cast<std::pair<std::wstring, std::wstring>*>(msg.lParam);
					SetWindowTextW(data.windowHandle, pair->first.c_str());
					SetWindowTextW(data.labelHandle, pair->second.c_str());
					delete pair;
				}
				break;
				case WM_HIDEPROGRESS:
				{
					closeTime = static_cast<DWORD>(msg.lParam);
				}
				break;
				}
			}
			Sleep(10);

			if (closeTime != 0 && closeTime > GetTickCount())
			{
				break;
			}
		}
		DestroyWindow(data.windowHandle);
		DeleteObject(data.labelFont);
		s_ProgressThreadId = 0;
		return 0;
	}

	void PlatformHelper::ShowProgressBar(const WString& title, const WString& info)
	{
		if (s_ProgressThreadId == 0)
		{
			CreateThread(nullptr, 0, ProgressUiThread, nullptr, 0, &s_ProgressThreadId);
			Sleep(5);
		}
		PostThreadMessage(s_ProgressThreadId, WM_SHOWPROGRESS, 0, (LPARAM)new std::pair(std::wstring(title), std::wstring(info)));
	}

	void PlatformHelper::HideProgressBar()
	{
		PostThreadMessage(s_ProgressThreadId, WM_HIDEPROGRESS, 0, GetTickCount() + 50);
	}

	String PlatformHelper::GetEditorDataFolder()
	{
		const char* adddataPath = getenv("APPDATA");
		return String(adddataPath).append("\\Blueberry\\");
	}
}