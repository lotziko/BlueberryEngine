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

		if (m_CurrentDirectory != Path::GetAssetsPath())
		{
			if (ImGui::Button("Back"))
			{
				m_CurrentDirectory = m_CurrentDirectory.parent_path();
			}
		}

		for (auto& it : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			const auto& path = it.path();
			auto pathString = path.string();
			auto extension = path.extension();
			auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());

			ImGui::PushID(pathString.c_str());
			if (it.is_directory())
			{
				if (ImGui::Button(relativePath.filename().string().c_str()))
				{
					m_CurrentDirectory /= path.filename();
				}
			}
			else if (extension == ".meta")
			{
				AssetImporter* importer = AssetDB::GetImporter(relativePath.string());
				if (importer != nullptr)
				{
					auto name = path.stem().string();
					if (ImGui::Button(name.c_str()))
					{
						Selection::SetActiveObject(importer);
					}

					if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
					{
						std::string stringPath = importer->GetFilePath();
						std::filesystem::path filePath = stringPath;
						if (filePath.extension() == ".scene")
						{
							EditorSceneManager::Load(stringPath);
						}
					}

					auto& objects = importer->GetImportedObjects();
					if (objects.size() > 0)
					{
						if (ImGui::BeginDragDropSource())
						{
							ImGui::SetDragDropPayload("OBJECT_ID", &objects[0], sizeof(ObjectId));
							ImGui::Text("%s", importer->GetName().c_str());
							ImGui::EndDragDropSource();
						}
					}
				}
			}
			ImGui::PopID();
		}

		const char* popupId = "ProjectPopup";
		if (ImGui::BeginPopup(popupId))
		{
			if (ImGui::MenuItem("Scene"))
			{
				EditorSceneManager::CreateEmpty("");
			}
			if (ImGui::MenuItem("Material"))
			{
				Material* material = Object::Create<Material>();
				AssetDB::CreateAsset(material, "Test.material");
				AssetDB::ImportAll();
			}
			ImGui::EndPopup();
		}
		
		if (mousePos.x >= pos.x && mousePos.y >= pos.y && mousePos.x <= pos.x + size.x && mousePos.y <= pos.y + size.y && ImGui::IsMouseClicked(1))
		{
			ImGui::OpenPopup(popupId);
		}

		ImGui::End();
	}
}
