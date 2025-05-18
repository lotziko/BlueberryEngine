#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct MenuTreeNode
	{
		String name;
		List<MenuTreeNode> children;
		void(*clickCallback)();
	};

	class MenuTree
	{
	public:
		MenuTree();

		const MenuTreeNode& GetRoot();
		void Add(const String& path, void(*clickCallback)());

	private:
		MenuTreeNode m_Root;
	};
}