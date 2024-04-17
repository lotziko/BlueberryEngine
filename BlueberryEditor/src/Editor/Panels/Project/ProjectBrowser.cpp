#include "bbpch.h"
#include "ProjectBrowser.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"
#include "Blueberry\Graphics\Material.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	ProjectBrowser::ProjectBrowser()
	{
		m_CurrentDirectory = Path::GetAssetsPath();
	}

	void ProjectBrowser::DrawUI()
	{
		ImGui::Begin("Project");

		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();

		auto assetsPath = Path::GetAssetsPath();

		if (m_CurrentDirectory != assetsPath)
		{
			if (ImGui::Button("Back"))
			{
				m_CurrentDirectory = m_CurrentDirectory.parent_path();
			}
		}

		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			const auto path = it.path();
			auto pathString = path.string();
			auto extension = path.extension();
			auto relativePath = std::filesystem::relative(path, assetsPath);

			ImGui::PushID(pathString.c_str());
			if (it.is_directory())
			{
				bool opened = ImGui::TreeNodeEx((void*)&pathString, ImGuiTreeNodeFlags_Leaf, relativePath.filename().string().c_str());
				if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
				{
					m_CurrentDirectory /= path.filename();
				}
				if (opened)
				{
					ImGui::TreePop();
				}
			}
			else if (extension == ".meta")
			{
				AssetImporter* importer = AssetDB::GetImporter(relativePath.replace_extension("").string());
				if (importer != nullptr)
				{
					auto name = path.stem().string();
					auto importedObjects = importer->GetImportedObjects();

					ImGuiTreeNodeFlags flags = importedObjects.size() > 0 ? ImGuiTreeNodeFlags_OpenOnArrow : ImGuiTreeNodeFlags_Leaf;
					bool opened = ImGui::TreeNodeEx((void*)importer, flags, name.c_str());

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

					if (opened)
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
					}
				}
			}
			ImGui::PopID();
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

		ImGui::End();
	}
}
