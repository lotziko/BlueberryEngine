#include "bbpch.h"
#include "SearchWindow.h"

#include "imgui\imgui_internal.h"

namespace Blueberry
{
	OBJECT_DEFINITION(EditorWindow, SearchWindow)

	void SearchWindow::Open(const Vector2& position)
	{
		SearchWindow* window = static_cast<SearchWindow*>(GetWindow(SearchWindow::Type));
		window->SetTitle("SearchWindow");
		window->m_Position = position;
		window->ShowPopup();
	}

	void SearchWindow::OnDrawUI()
	{
		ImGui::SetWindowPos(ImVec2(m_Position.x, m_Position.y));

		if (!ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
		{
			Close();
		}
	}
}
