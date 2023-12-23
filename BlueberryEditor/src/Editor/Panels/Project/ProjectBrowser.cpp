#include "bbpch.h"
#include "ProjectBrowser.h"

#include "Editor\Path.h"
#include "Editor\Serialization\AssetImporter.h"
#include "Editor\Serialization\AssetDB.h"
#include "Editor\EditorSceneManager.h"
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
				AssetImporter* importer = AssetDB::GetImporter(pathString);
				if (importer != nullptr)
				{
					auto name = importer->GetName().c_str();
					if (ImGui::Button(name))
					{
						// TODO make it select AssetImporter
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
							ImGui::Text("%s", name);
							ImGui::EndDragDropSource();
						}
					}
				}
			}
			ImGui::PopID();
		}

		ImGui::End();
	}
}
