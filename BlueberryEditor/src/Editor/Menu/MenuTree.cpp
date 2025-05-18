#include "MenuTree.h"

#include "Blueberry\Tools\StringHelper.h"

namespace Blueberry
{
	MenuTree::MenuTree()
	{
		m_Root.name = "Root";
	}

	const MenuTreeNode& MenuTree::GetRoot()
	{
		return m_Root;
	}

	void MenuTree::Add(const String& path, void(*clickCallback)())
	{
		List<String> splittedPath;
		StringHelper::Split(path.c_str(), '/', splittedPath);
		MenuTreeNode* currentNode = &m_Root;
		for (auto pathIt = splittedPath.begin(); pathIt < splittedPath.end(); ++pathIt)
		{
			bool hasNext = false;
			for (auto it = currentNode->children.begin(); it < currentNode->children.end(); ++it)
			{
				if (it->name == *pathIt)
				{
					currentNode = it._Ptr;
					hasNext = true;
					break;
				}
			}
			if (hasNext)
			{
				continue;
			}
			else
			{
				MenuTreeNode newNode = {};
				newNode.name = *pathIt;
				currentNode = &currentNode->children.emplace_back(newNode);

				if (pathIt == splittedPath.end() - 1)
				{
					currentNode->clickCallback = clickCallback;
				}
			}
		}
	}
}
