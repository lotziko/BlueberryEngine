#include "bbpch.h"
#include "FolderTree.h"

namespace Blueberry
{
	const FolderTreeNode& FolderTree::GetRoot()
	{
		return m_Root;
	}

	void FolderTree::Update(const std::string& root)
	{
		std::filesystem::path path = root;
		m_Root = {};
		m_Root.path = path;
		m_Root.name = path.filename().string();
		Populate(&m_Root);
	}

	void FolderTree::Populate(FolderTreeNode* parent)
	{
		for (auto& it : std::filesystem::directory_iterator(parent->path))
		{
			if (it.is_directory())
			{
				std::filesystem::path path = it;
				FolderTreeNode child;
				child.path = path;
				child.name = path.filename().string();
				Populate(&child);
				parent->children.emplace_back(child);
			}
		}
	}
}
