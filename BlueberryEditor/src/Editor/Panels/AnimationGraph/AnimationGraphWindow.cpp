#include "AnimationGraphWindow.h"

#include "Blueberry\Core\ClassDB.h"
#include "Editor\Menu\EditorMenuManager.h"

#include <imgui\imgui.h>
#include <imguinode\imgui_node_editor.h>

namespace Blueberry
{
	OBJECT_DEFINITION(AnimationGraphWindow, EditorWindow)
	{
		DEFINE_BASE_FIELDS(AnimationGraphWindow, EditorWindow)
		EditorMenuManager::AddItem("Window/AnimationGraph", &AnimationGraphWindow::Open);
	}

	void AnimationGraphWindow::Open()
	{
		EditorWindow* window = GetWindow(AnimationGraphWindow::Type);
		window->SetTitle("Animation Graph");
		window->Show();
	}

	void AnimationGraphWindow::OnDrawUI()
	{
	}
}
