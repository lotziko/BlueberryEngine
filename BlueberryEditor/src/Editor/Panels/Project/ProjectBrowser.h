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
		void DrawFile(const std::filesystem::path& path);

		void UpdateFiles();

	private:
		std::filesystem::path m_PreviousDirectory;
		std::filesystem::path m_CurrentDirectory;
		std::vector<std::filesystem::path> m_CurrentDirectoryFiles;

		FolderTree m_FolderTree;
		const char* m_OpenedModalPopupId = nullptr;

		Texture2D* m_FolderIcon;
		Texture2D* m_FolderIconSmall;
		Texture2D* m_FolderIconSmallOpened;
		Texture2D* m_FbxIcon;
	};
}