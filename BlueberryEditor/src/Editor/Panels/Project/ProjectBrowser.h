#pragma once

#include "FolderTree.h"
#include <filesystem>

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class Texture2D;
	class AssetImporter;
	class Object;
	class WindowDropFilesEventArgs;

	class ProjectBrowser : public EditorWindow
	{
		OBJECT_DECLARATION(ProjectBrowser)

		struct AssetInfo
		{
			std::filesystem::path path;
			String pathString;
			AssetImporter* importer;
			List<ObjectPtr<Object>> objects; // First object is main
			List<Vector2> positions;
			bool isDirectory;
			bool expanded;
		};

	public:
		ProjectBrowser();
		virtual ~ProjectBrowser();

		static void Open();
		static void ShowObject(Object* object);

		virtual void OnDrawUI() final;

	private:
		void DrawFoldersTree();
		void DrawFolderNode(const FolderTreeNode& node);
		void DrawCurrentFolder();
		void DrawObject(Object* object, const AssetInfo& asset, bool& anyHovered);
		void OpenAsset(const AssetInfo& asset);

		void OnAssetDBRefresh();
		void OnWindowDropFiles(const WindowDropFilesEventArgs& args);

		void UpdateTree();
		void UpdateFiles();

	private:
		std::filesystem::path m_PreviousDirectory;
		std::filesystem::path m_CurrentDirectory;
		List<AssetInfo> m_CurrentDirectoryAssets;
		Object* m_ScrollRequest = nullptr;
		Object* m_RenamingObject = nullptr;

		Object* m_CreatingObject = nullptr;
		String m_CreatingObjectName;

		FolderTree m_FolderTree;
		const char* m_OpenedModalPopupId = nullptr;

		float m_FoldersColumnWidth = 200;

		Texture2D* m_FolderIconSmall;
		Texture2D* m_FolderIconSmallOpened;
	};
}