#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	enum DialogResult
	{
		Yes,
		No,
		Cancel
	};

	class PlatformHelper
	{
	public:
		static WString OpenFileDialog();
		static DialogResult OpenDialog(const WString& titleText, const WString& contentText, const WString& yesText, const WString& noText, const WString& cancelText);
		static void RevealInExplorer(const WString& path);
		static void ShowProgressBar(const WString& title, const WString& info);
		static void HideProgressBar();
		static String GetEditorDataFolder();
		static void MoveToRecycleBin(const WString& path);
	};
}