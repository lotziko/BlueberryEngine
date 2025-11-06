#pragma once

#include "Blueberry\Core\Base.h"
#include <filesystem>

namespace Blueberry
{
	struct FolderTreeNode
	{
		String name;
		std::filesystem::path path;
		List<FolderTreeNode> children;
	};

	class FolderTree
	{
	public:
		FolderTree() = default;

		const FolderTreeNode& GetRoot();
		void Update(const String& root);

	private:
		void Populate(FolderTreeNode* parent);

	private:
		FolderTreeNode m_Root;
	};
}