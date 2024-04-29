#pragma once

#include "FolderTree.h"
#include <filesystem>

namespace Blueberry
{
	class Texture2D;

	class ProjectBrowser
	{
	public:
		ProjectBrowser();
		virtual ~ProjectBrowser() = default;

		void DrawUI();

	private:
		void DrawFoldersTree();
		void DrawFolderNode(const FolderTreeNode& node);
		void DrawCurrentFolder();
		void DrawFile(const std::filesystem::directory_entry& file);

	private:
		std::filesystem::path m_CurrentDirectory;
		FolderTree m_FolderTree;
		const char* m_OpenedModalPopupId = nullptr;
		Texture2D* m_FolderIcon;
		Texture2D* m_FbxIcon;
	};
}