#pragma once

#include "MenuTree.h"

namespace Blueberry
{
	class EditorMenuManager
	{
	public:
		static const MenuTreeNode& GetRoot();
		static void AddItem(const String& path, void(*clickCallback)());

	private:
		static MenuTree s_Tree;
	};
}