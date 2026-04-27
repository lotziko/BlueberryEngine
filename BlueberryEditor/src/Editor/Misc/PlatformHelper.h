#pragma once

#include "Blueberry\Core\Base.h"

#include <filesystem>

namespace Blueberry
{
	enum class DialogResult
	{
		Yes,
		No,
		Cancel
	};

	class PlatformHelper
	{
	public:
		static String OpenFileDialog();
		static DialogResult OpenDialog(const String& titleText, const String& contentText, const String& yesText, const String& noText, const String& cancelText);
		static void RevealInExplorer(const String& path);
		static void ShowProgressBar(const String& title, const String& info);
		static void HideProgressBar();
		static String GetEditorDataFolder();
		static void MoveToRecycleBin(const String& path);
	};
}