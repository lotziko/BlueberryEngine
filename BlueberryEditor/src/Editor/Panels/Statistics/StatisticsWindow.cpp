#include "StatisticsWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Logging\Profiler.h"
#include "Editor\Menu\EditorMenuManager.h"

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
		for (auto& pair : Profiler::GetData())
		{
			ImGui::Text("%s %f milliseconds", pair.first, pair.second);
		}
	}
}
