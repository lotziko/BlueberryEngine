#include "bbpch.h"
#include "ProjectBrowser.h"

#include "Editor\Path.h"
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
					
				}
			}
		}

		ImGui::End();
	}
}
