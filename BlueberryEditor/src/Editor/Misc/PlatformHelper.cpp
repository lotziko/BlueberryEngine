#include "PlatformHelper.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Window.h"
#include "Blueberry\Tools\StringHelper.h"

#include <shobjidl.h> // IFileOpenDialog
#include <shlobj.h> // SHOpenFolderAndSelectItems
#include <commctrl.h>
#include <filesystem>

#include <atomic>

// https://cybernaught.net/code/codeAPItaskdialog.html
#pragma comment(lib, "comctl32.lib")
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace Blueberry
{
	static DWORD s_ProgressThreadId = 0;
	static HANDLE s_ProgressThreadReadyEvent = 0;

	#define WM_SHOWPROGRESS (WM_USER + 1)
	#define WM_HIDEPROGRESS (WM_USER + 2)

	String PlatformHelper::OpenFileDialog()
	{
		String result;
		IFileOpenDialog* dialog = nullptr;

		HRESULT hr = CoCreateInstance(
			CLSID_FileOpenDialog,
			nullptr,
			CLSCTX_INPROC_SERVER,
			IID_PPV_ARGS(&dialog)
		);

		if (FAILED(hr))
		{
			return "";
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
					result = StringHelper::WideToString(path);
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

	DialogResult PlatformHelper::OpenDialog(const String& titleText, const String& contentText, const String& yesText, const String& noText, const String& cancelText)
	{
		TASKDIALOG_BUTTON buttons[] =
		{
			{ IDYES, StringHelper::StringToWide(yesText).c_str() },
			{ IDNO, StringHelper::StringToWide(noText).c_str() },
			{ IDCANCEL, StringHelper::StringToWide(cancelText).c_str() }
		};

		WString wtitleText = StringHelper::StringToWide(titleText);
		WString wcontentText = StringHelper::StringToWide(contentText);

		TASKDIALOGCONFIG config = {};
		config.cbSize = sizeof(config);
		config.hwndParent = NULL;
		config.dwFlags = TDF_ALLOW_DIALOG_CANCELLATION;
		config.dwCommonButtons = 0;
		config.pszWindowTitle = wtitleText.c_str();
		config.pszMainIcon = TD_WARNING_ICON;
		config.pszMainInstruction = NULL;
		config.pszContent = wcontentText.c_str();
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

	void PlatformHelper::RevealInExplorer(const String& path)
	{
		PIDLIST_ABSOLUTE pidl = nullptr;
		HRESULT hr = SHParseDisplayName(
			StringHelper::StringToWide(path).c_str(),
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

	DWORD WINAPI ProgressUiThread(void*)
	{
		BB_INITIALIZE_ALLOCATOR_THREAD();
		WNDCLASSW wc{};
		wc.lpfnWndProc = ProgressWndProc;
		wc.hInstance = GetModuleHandle(nullptr);
		wc.lpszClassName = L"ProgressWindow";
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);

		RegisterClassW(&wc);

		HWND windowHandle = CreateWindowExW(
			WS_EX_DLGMODALFRAME | WS_EX_DLGMODALFRAME | WS_EX_NOACTIVATE,
			L"ProgressWindow",
			NULL,
			WS_CAPTION,
			CW_USEDEFAULT, CW_USEDEFAULT, 540, 160,
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
			20, 50, 480, 20,
			windowHandle,
			nullptr,
			GetModuleHandle(nullptr),
			nullptr
		);
		SendMessage(progressBarHandle, PBM_SETMARQUEE, TRUE, 0);

		HWND labelHandle = CreateWindowEx(
			0,
			L"STATIC",
			L"Test",
			WS_CHILD | WS_VISIBLE | SS_LEFT,
			20, 80, 480, 20,
			windowHandle,
			nullptr,
			GetModuleHandle(nullptr),
			nullptr
		);

		NONCLIENTMETRICS ncm = { sizeof(NONCLIENTMETRICS) };
		SystemParametersInfo(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0);
		HFONT labelFont = CreateFontIndirect(&ncm.lfCaptionFont);
		SendMessage(labelHandle, WM_SETFONT, (WPARAM)labelFont, TRUE);

		MSG msg;
		ULONGLONG showTime = 0;
		ULONGLONG hideTime = 0;
		INT counter = 0;
		BOOL isVisible = false;
		PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
		SetEvent(s_ProgressThreadReadyEvent);

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
					++counter;
					if (!isVisible && counter > 0)
					{
						isVisible = true;
						showTime = GetTickCount64() + 10;
					}
					hideTime = 0;
					auto pair = reinterpret_cast<std::pair<WString, WString>*>(msg.lParam);
					SetWindowTextW(windowHandle, pair->first.c_str());
					SetWindowTextW(labelHandle, pair->second.c_str());
					delete pair;
				}
				break;
				case WM_HIDEPROGRESS:
				{
					--counter;
					showTime = 0;
					if (isVisible && counter <= 0)
					{
						isVisible = false;
						hideTime = GetTickCount64() + 10;
					}
				}
				break;
				}
			}
			Sleep(5);

			ULONGLONG tickCount = GetTickCount64();
			if (showTime != 0 && tickCount > showTime)
			{
				ShowWindow(windowHandle, SW_SHOW);
				SetForegroundWindow(windowHandle);
				showTime = 0;
			}
			if (hideTime != 0 && tickCount > hideTime)
			{
				ShowWindow(windowHandle, SW_HIDE);
				hideTime = 0;
			}
		}
		DestroyWindow(windowHandle);
		DeleteObject(labelFont);
		s_ProgressThreadId = 0;
		BB_SHUTDOWN_ALLOCATOR_THREAD();
		return 0;
	}

	void PlatformHelper::ShowProgressBar(const String& title, const String& info)
	{
		if (s_ProgressThreadId == 0)
		{
			s_ProgressThreadReadyEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
			CreateThread(nullptr, 0, ProgressUiThread, nullptr, 0, &s_ProgressThreadId);
			if (s_ProgressThreadReadyEvent != 0)
			{
				WaitForSingleObject(s_ProgressThreadReadyEvent, INFINITE);
				CloseHandle(s_ProgressThreadReadyEvent);
			}
		}
		PostThreadMessage(s_ProgressThreadId, WM_SHOWPROGRESS, 0, (LPARAM)new std::pair(StringHelper::StringToWide(title), StringHelper::StringToWide(info)));
	}

	void PlatformHelper::HideProgressBar()
	{
		PostThreadMessage(s_ProgressThreadId, WM_HIDEPROGRESS, 0, 0);
	}

	String PlatformHelper::GetEditorDataFolder()
	{
		const char* adddataPath = getenv("APPDATA");
		return String(adddataPath).append("\\Blueberry\\");
	}

	void PlatformHelper::MoveToRecycleBin(const String& path)
	{
		WString wpath = StringHelper::StringToWide(path);
		SHFILEOPSTRUCTW fileOp = { 0 };
		fileOp.hwnd = NULL;
		fileOp.wFunc = FO_DELETE;
		fileOp.pFrom = wpath.c_str();
		fileOp.pTo = NULL;
		fileOp.fFlags = FOF_ALLOWUNDO | FOF_NOERRORUI | FOF_NOCONFIRMATION | FOF_SILENT;
		int result = SHFileOperationW(&fileOp);
		if (result != 0)
		{
			BB_ERROR("Failed to delete file " << path.c_str());
		}
	}
}