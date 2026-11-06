#include "StatisticsWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Logging\Profiler.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Blueberry\Graphics\Concrete\GraphicsSettings.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	OBJECT_DEFINITION(StatisticsWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(StatisticsWindow, EditorWindow)
		EditorMenuManager::AddItem("Window/Statistics", &StatisticsWindow::Open);
	}

	void StatisticsWindow::Open()
	{
		EditorWindow* window = GetWindow(StatisticsWindow::Type);
		window->SetTitle("Statistics");
		window->Show();
	}

	void StatisticsWindow::OnDrawUI()
	{
		bool screenReflections = GraphicsSettings::GetScreenReflections();
		if (ImGui::Checkbox("Screen Reflections", &screenReflections))	// TODO move to Debug window
		{
			GraphicsSettings::SetScreenReflections(screenReflections);
		}

		ImGui::EditorStyle& style = ImGui::GetEditorStyle();
		ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(10, 5));
		for (auto& pair : Profiler::GetData())
		{
			ImGui::Text("%s %f milliseconds", pair.first, pair.second);
		}
		ImGui::EndPaddedArea();
	}
}
