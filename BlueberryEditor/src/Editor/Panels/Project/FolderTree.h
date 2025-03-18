#pragma once

#include <filesystem>

namespace Blueberry
{
	struct FolderTreeNode
	{
		std::string name;
		std::filesystem::path path;
		List<FolderTreeNode> children;
	};

	class FolderTree
	{
	public:
		FolderTree() = default;

		const FolderTreeNode& GetRoot();
		void Update(const std::string& root);

	private:
		void Populate(FolderTreeNode* parent);

	private:
		FolderTreeNode m_Root;
	};
}