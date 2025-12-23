#include "PlatformHelper.h"

#include <shobjidl.h> // IFileOpenDialog
#include <shlobj.h> // SHOpenFolderAndSelectItems
#include <commctrl.h>

// https://cybernaught.net/code/codeAPItaskdialog.html
#pragma comment(lib, "comctl32.lib")
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace Blueberry
{
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

	String PlatformHelper::GetEditorDataFolder()
	{
		const char* adddataPath = getenv("APPDATA");
		return String(adddataPath).append("\\Blueberry\\");
	}
}