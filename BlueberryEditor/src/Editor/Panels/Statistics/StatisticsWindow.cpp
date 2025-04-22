#include "bbpch.h"
#include "StatisticsWindow.h"

#include "imgui\imgui.h"

#include "Editor\Menu\EditorMenuManager.h"

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
