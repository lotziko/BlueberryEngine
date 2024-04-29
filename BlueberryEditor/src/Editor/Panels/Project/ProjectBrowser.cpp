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
		ImGuiTreeNodeFlags flags = node.children.size() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;
		bool opened = ImGui::TreeNodeEx(node.name.c_str(), flags);

		if (ImGui::IsItemClicked())
		{
			m_CurrentDirectory = node.path;
		}

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
		//ImGui::NewLine();
		/*if (m_CurrentDirectory != Path::GetAssetsPath())
		{
			if (ImGui::Button("Back"))
			{
				m_CurrentDirectory = m_CurrentDirectory.parent_path();
			}
			ImGui::NewLine();
		}*/

		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			DrawFile(it);
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

				Material* material = Object::Create<Material>();
				AssetDB::CreateAsset(material, materialName);
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

	void ProjectBrowser::DrawFile(const std::filesystem::directory_entry& file)
	{
		const int cellSize = 90;
		const int cellIconPadding = 8;

		const auto path = file.path();
		auto pathString = path.string();
		auto extension = path.extension();
		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());

		ImGui::PushID(pathString.c_str());
		ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 1.0f));
		if (file.is_directory())
		{
			ImVec2 pos = ImGui::GetCursorScreenPos();
			ImGui::Selectable(relativePath.filename().string().c_str(), false, 0, ImVec2(cellSize, cellSize));
			ImGui::GetWindowDrawList()->AddImage(m_FolderIcon->GetHandle(), ImVec2(pos.x + cellIconPadding, pos.y), ImVec2(pos.x + cellSize - cellIconPadding, pos.y + cellSize - cellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

			if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			{
				m_CurrentDirectory /= path.filename();
			}
		}
		else if (extension == ".meta")
		{
			AssetImporter* importer = AssetDB::GetImporter(relativePath.replace_extension("").string());
			if (importer != nullptr)
			{
				auto name = path.stem().string();
				auto importedObjects = importer->GetImportedObjects();

				//https://www.google.com/search?q=imgui+selectable&oq=imgui+selectable&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhB0gEIMjA0NGowajGoAgCwAgA&sourceid=chrome&ie=UTF-8
				ImVec2 pos = ImGui::GetCursorScreenPos();
				ImGui::Selectable(name.c_str(), false, 0, ImVec2(cellSize, cellSize));
				ImGui::GetWindowDrawList()->AddImage(m_FbxIcon->GetHandle(), ImVec2(pos.x + cellIconPadding, pos.y), ImVec2(pos.x + cellSize - cellIconPadding, pos.y + cellSize - cellIconPadding * 2), ImVec2(0, 1), ImVec2(1, 0));

				if (ImGui::IsItemHovered())
				{
					importer->ImportDataIfNeeded();

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

				/*if (opened)
				{
					for (auto& pair : importedObjects)
					{
						Object* object = ObjectDB::GetObject(pair.second);
						bool opened = ImGui::TreeNodeEx((void*)object, ImGuiTreeNodeFlags_Leaf, object->GetName().c_str());

						if (ImGui::BeginDragDropSource())
						{
							ObjectId objectId = object->GetObjectId();
							ImGui::SetDragDropPayload("OBJECT_ID", &objectId, sizeof(ObjectId));
							ImGui::Text("%s", importer->GetName().c_str());
							ImGui::EndDragDropSource();
						}

						if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
						{
							Selection::SetActiveObject(importer);
						}

						if (opened)
						{
							ImGui::TreePop();
						}
					}
					ImGui::TreePop();
				}*/
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
}
