#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"
#include <filesystem>

namespace Blueberry
{
	struct FolderTreeNode
	{
		String name;
		std::filesystem::path path;
		Guid guid;
		int depth = 0;
	};

	class FolderTree
	{
	public:
		FolderTree() = default;

		List<FolderTreeNode>& GetNodes();
		void Update(const std::filesystem::path& rootPath);

	private:
		void Populate(const std::filesystem::path& parentPath, int depth);

	private:
		List<FolderTreeNode> m_Nodes;
	};
}