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

			if (it.is_directory())
			{
				if (ImGui::Button(relativePath.filename().string().c_str()))
				{
					m_CurrentDirectory /= path.filename();
				}
			}
			else if (extension == ".meta")
			{
				if (ImGui::Button(relativePath.stem().string().c_str()))
				{
					// TODO make it select AssetImporter
				}

				if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
				{
					AssetImporter* importer = AssetDB::GetImporter(pathString);
					if (importer != nullptr)
					{
						std::string stringPath = importer->GetFilePath();
						std::filesystem::path filePath = stringPath;
						if (filePath.extension() == ".scene")
						{
							EditorSceneManager::Load(stringPath);
						}
					}
				}

				/*if (ImGui::BeginDragDropSource())
				{
					ImGui::SetDragDropPayload("ASSET_OBJECT", &path, sizeof(std::filesystem::path));
					ImGui::Text("%ws", relativePath.replace_extension().c_str());
					ImGui::EndDragDropSource();
				}*/
			}
		}

		ImGui::End();
	}
}
