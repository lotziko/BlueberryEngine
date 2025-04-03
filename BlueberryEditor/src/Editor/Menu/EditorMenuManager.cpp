#include "bbpch.h"
#include "EditorMenuManager.h"

namespace Blueberry
{
	MenuTree EditorMenuManager::s_Tree = {};

	const MenuTreeNode& EditorMenuManager::GetRoot()
	{
		return s_Tree.GetRoot();
	}

	void EditorMenuManager::AddItem(const std::string& path, void(*clickCallback)())
	{
		s_Tree.Add(path, clickCallback);
	}
}
