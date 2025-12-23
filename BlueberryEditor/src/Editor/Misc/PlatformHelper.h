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
		static String GetEditorDataFolder();
	};
}