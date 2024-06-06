#include "bbpch.h"
#include "ProjectBrowser.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	ProjectBrowser::ProjectBrowser()
	{
		m_CurrentDirectory = Path::GetAssetsPath();

		m_FolderIcon = (Texture2D*)AssetLoader::Load("assets/icons/FolderIcon.png");
		m_FolderIconSmall = (Texture2D*)AssetLoader::Load("assets/icons/FolderIconSmall.png");
		m_FolderIconSmallOpened = (Texture2D*)AssetLoader::Load("assets/icons/FolderIconSmallOpened.png");
		m_FbxIcon = (Texture2D*)AssetLoader::Load("assets/icons/FbxIcon.png");

		m_FolderTree = FolderTree(m_CurrentDirectory.string());
	}

	void ProjectBrowser::DrawUI()
	{
		ImGui::Begin("Project");

		ImGuiTableFlags tableFlags = 0;
		tableFlags |= ImGuiTableFlags_Resizable;

		if (ImGui::BeginTable("##ProjectTable", 2, tableFlags))
		{
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);

			ImGui::BeginChild("##FoldersColumn", ImVec2(0, ImGui::GetContentRegionAvail().y - 2));
			DrawFoldersTree();
			ImGui::EndChild();

			ImGui::TableSetColumnIndex(1);
			ImGui::BeginChild("##CurrentFolderColumn", ImVec2(0, ImGui::GetContentRegionAvail().y - 2));
			DrawCurrentFolder();
			ImGui::EndChild();

			ImGui::EndTable();
		}
		
		ImGui::End();
	}

	void ProjectBrowser::DrawFoldersTree()
	{
		DrawFolderNode(m_FolderTree.GetRoot());
	}

	void ProjectBrowser::DrawFolderNode(const FolderTreeNode& node)
	{
		ImGuiTreeNodeFlags flags = (node.children.size() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf) | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth;
		ImVec2 pos = ImGui::GetCursorScreenPos();
		const char* label = node.name.c_str();
		bool opened = ImGui::TreeNodeEx(label, flags, "");

		if (!ImGui::IsItemToggledOpen() && ImGui::IsItemClicked())
		{
			m_CurrentDirectory = node.path;
		}

		ImGui::SameLine();
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4);
		ImGui::Image((opened && node.children.size() > 0) ? m_FolderIconSmallOpened->GetHandle() : m_FolderIconSmall->GetHandle(), ImVec2(16, 16), ImVec2(0, 1), ImVec2(1, 0));
		ImGui::SameLine();
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4);
		ImGui::Text("%s", label);

		if (opened)
		{
			for (auto& child : node.children)
			{
				DrawFolderNode(child);
			}
			ImGui::TreePop();
		}
	}

	void ProjectBrowser::DrawCurrentFolder()
	{
		bool isAnyFileHovered = false;

		if (m_CurrentDirectory != m_PreviousDirectory)
		{
			UpdateFiles();
		}
		
		for (auto& path : m_CurrentDirectoryFiles)
		{
			DrawFile(path);
			if (ImGui::IsItemHovered())
			{
				isAnyFileHovered = true;
			}
		}

		if (!isAnyFileHovered && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
		{
			Selection::SetActiveObject(nullptr);
		}

		const char* popId = "Delete?";
		const char* popupId = "ProjectPopup";
		if (ImGui::BeginPopup(popupId))
		{
			if (ImGui::MenuItem("Scene"))
			{
				EditorSceneManager::CreateEmpty("");
			}
			if (ImGui::MenuItem("Material"))
			{
				m_OpenedModalPopupId = popId;
			}

			ImGui::EndPopup();
		}

		if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1))
		{
			ImGui::OpenPopup(popupId);
		}

		if (m_OpenedModalPopupId == popId)
		{
			ImGui::OpenPopup(popId);
		}
		if (ImGui::BeginPopupModal(popId, NULL, ImGuiWindowFlags_AlwaysAutoResize))
		{
			static char name[256];
			ImGui::InputText("Name", name, 256);
			ImGui::Separator();

			if (ImGui::Button("OK", ImVec2(120, 0)))
			{
				std::string materialName(name);
				materialName.append(".material");

				auto relativePath = std::filesystem::relative(m_CurrentDirectory, Path::GetAssetsPath());
				relativePath.append(materialName);

				Material* material = Object::Create<Material>();
				AssetDB::CreateAsset(material, relativePath.string());
				AssetDB::SaveAssets();
				AssetDB::Refresh();

				m_OpenedModalPopupId = nullptr;
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		if (ImGui::IsKeyPressed(ImGuiKey_S) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
		{
			AssetDB::SaveAssets();
		}
	}

	void ProjectBrowser::DrawFile(const std::filesystem::path& path)
	{
		const int cellSize = 90;
		const int cellIconPadding = 8;

		auto pathString = path.string();
		auto extension = path.extension();
		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());

		ImGui::PushID(pathString.c_str());
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));

		AssetImporter* importer = AssetDB::GetImporter(relativePath.string());
		if (importer != nullptr)
		{
			auto name = path.stem().string();
			auto importedObjects = importer->GetImportedObjects();
			bool isDirectory = std::filesystem::is_directory(path);

			//https://www.google.com/search?q=imgui+selectable&oq=imgui+selectable&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhB0gEIMjA0NGowajGoAgCwAgA&sourceid=chrome&ie=UTF-8
			ImVec2 pos = ImGui::GetCursorScreenPos();
			if (ImGui::Selectable(name.c_str(), Selection::IsActiveObject(importer), 0, ImVec2(cellSize, cellSize)))
			{
				if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
				{
					Selection::AddActiveObject(importer);
				}
				else
				{
					Selection::SetActiveObject(importer);
				}
			}
			ImGui::GetWindowDrawList()->AddImage(isDirectory ? m_FolderIcon->GetHandle() : m_FbxIcon->GetHandle(), ImVec2(pos.x + cellIconPadding, pos.y), ImVec2(pos.x + cellSize - cellIconPadding, pos.y + cellSize - cellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

			if (ImGui::IsItemHovered())
			{
				if (isDirectory)
				{
					if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
					{
						std::filesystem::path directoryPath = importer->GetFilePath();
						m_CurrentDirectory /= directoryPath.filename();
					}
				}
				else
				{
					if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
					{
						std::string stringPath = importer->GetFilePath();
						std::filesystem::path filePath = stringPath;
						if (filePath.extension() == ".scene")
						{
							EditorSceneManager::Load(stringPath);
						}
					}
				}
			}

			// TODO main object, selection into OBJECT_ID and dropdown for multiple objects
			if (ImGui::BeginDragDropSource())
			{
				auto it = importer->GetImportedObjects().find(importer->GetMainObject());
				if (it != importer->GetImportedObjects().end())
				{
					ObjectId objectId = it->second;
					ImGui::SetDragDropPayload("OBJECT_ID", &objectId, sizeof(ObjectId));
					ImGui::Text("%s", importer->GetName().c_str());
				}
				ImGui::EndDragDropSource();
			}
		}

		float visibleWidth = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
		if (ImGui::GetItemRectMax().x + cellSize < visibleWidth)
		{
			ImGui::SameLine();
		}

		ImGui::PopID();
		ImGui::PopStyleVar();
	}

	void ProjectBrowser::UpdateFiles()
	{
		m_PreviousDirectory = m_CurrentDirectory;
		m_CurrentDirectoryFiles.clear();

		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			auto path = it.path();
			if (it.is_directory() && path.extension() != ".meta")
			{
				m_CurrentDirectoryFiles.emplace_back(path);
			}
		}
		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			auto path = it.path();
			if (!it.is_directory() && path.extension() != ".meta")
			{
				m_CurrentDirectoryFiles.emplace_back(path);
			}
		}
	}
}
