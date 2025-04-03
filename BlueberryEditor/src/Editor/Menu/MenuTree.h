#pragma once

namespace Blueberry
{
	struct MenuTreeNode
	{
		std::string name;
		List<MenuTreeNode> children;
		void(*clickCallback)();
	};

	class MenuTree
	{
	public:
		MenuTree();

		const MenuTreeNode& GetRoot();
		void Add(const std::string& path, void(*clickCallback)());

	private:
		MenuTreeNode m_Root;
	};
}