#include "DebugWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Blueberry\Graphics\Concrete\GraphicsSettings.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	OBJECT_DEFINITION(DebugWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(DebugWindow, EditorWindow)
		EditorMenuManager::AddItem("Window/Debug", &DebugWindow::Open);
	}

	void DebugWindow::Open()
	{
		EditorWindow* window = GetWindow(DebugWindow::Type);
		window->SetTitle("Debug");
		window->Show();
	}

	void DebugWindow::OnDrawUI()
	{
		ImGui::EditorStyle& style = ImGui::GetEditorStyle();
		ImGui::BeginPaddedArea(ImVec2(10, 5), ImVec2(10, 5));

		bool screenReflections = GraphicsSettings::GetScreenReflections();
		if (ImGui::BoolEdit("Screen Reflections", &screenReflections))
		{
			GraphicsSettings::SetScreenReflections(screenReflections);
		}

		float debugValue0 = GraphicsSettings::GetDebugValue0();
		if (ImGui::FloatEdit("Debug value 0", &debugValue0, 0, 1))
		{
			GraphicsSettings::SetDebugValue0(debugValue0);
		}

		float debugValue1 = GraphicsSettings::GetDebugValue1();
		if (ImGui::FloatEdit("Debug value 1", &debugValue1, 0, 1))
		{
			GraphicsSettings::SetDebugValue1(debugValue1);
		}

		ImGui::EndPaddedArea();
	}
}
