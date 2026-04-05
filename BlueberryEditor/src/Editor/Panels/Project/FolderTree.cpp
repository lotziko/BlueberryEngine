#include "FolderTree.h"

#include "Blueberry\Tools\StringHelper.h"

namespace Blueberry
{
	const FolderTreeNode& FolderTree::GetRoot()
	{
		return m_Root;
	}

	void FolderTree::Update(const String& root)
	{
		std::filesystem::path path = root;
		m_Root = {};
		m_Root.path = path;
		m_Root.name = StringHelper::ToString(path.filename());
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
				child.name = StringHelper::ToString(path.filename());
				Populate(&child);
				parent->children.push_back(child);
			}
		}
	}
}
