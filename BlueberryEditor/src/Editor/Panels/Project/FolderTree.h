#pragma once

#include <filesystem>

namespace Blueberry
{
	struct FolderTreeNode
	{
		std::string name;
		std::filesystem::path path;
		std::vector<FolderTreeNode> children;
	};

	class FolderTree
	{
	public:
		FolderTree() = default;
		FolderTree(const std::string& root);

		const FolderTreeNode& GetRoot();

	private:
		void Populate(FolderTreeNode* parent);

	private:
		FolderTreeNode m_Root;
	};
}