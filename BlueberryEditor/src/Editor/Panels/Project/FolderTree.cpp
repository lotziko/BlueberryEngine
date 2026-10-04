#include "FolderTree.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\AssetImporter.h"

#include "Blueberry\Tools\StringHelper.h"

namespace Blueberry
{
	List<FolderTreeNode>& FolderTree::GetNodes()
	{
		return m_Nodes;
	}

	void FolderTree::Update(const std::filesystem::path& rootPath)
	{
		m_Nodes.clear();
		FolderTreeNode rootNode = {};
		rootNode.path = rootPath;
		rootNode.name = StringHelper::ToString(rootPath.filename());
		rootNode.guid = {};
		rootNode.depth = 0;
		m_Nodes.push_back(std::move(rootNode));
		Populate(rootPath, 1);
	}

	void FolderTree::Populate(const std::filesystem::path& parentPath, int depth)
	{
		for (auto& it : std::filesystem::directory_iterator(parentPath))
		{
			if (it.is_directory())
			{
				std::filesystem::path path = it;
				Guid guid = {};
				String relativePath = StringHelper::ToString(std::filesystem::relative(path, Path::GetAssetsPath()));
				AssetImporter* importer = AssetDB::GetImporter(relativePath);
				if (importer != nullptr)
				{
					guid = importer->GetGuid();
				}

				FolderTreeNode child;
				child.path = path;
				child.name = StringHelper::ToString(path.filename());
				child.guid = guid;
				child.depth = depth;
				m_Nodes.push_back(std::move(child));
				Populate(path, depth + 1);
			}
		}
	}
}
