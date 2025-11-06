#include "SceneInspector.h"

#include "InspectorExpandedItemsCache.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Inspector\ObjectEditor.h"
#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Selection.h"
#include "Editor\Menu\EditorMenuManager.h"
#include "Editor\Serialization\SerializedObject.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	OBJECT_DEFINITION(SceneInspector, EditorWindow)
	{
		DEFINE_BASE_FIELDS(SceneInspector, EditorWindow)
		EditorMenuManager::AddItem("Window/Inspector", &SceneInspector::Open);
	}

	SceneInspector::SceneInspector()
	{
		InspectorExpandedItemsCache::Load();
		Selection::GetSelectionChanged().AddCallback<SceneInspector, &SceneInspector::SelectionChanged>(this);
	}

	SceneInspector::~SceneInspector()
	{
		InspectorExpandedItemsCache::Save();
		Selection::GetSelectionChanged().RemoveCallback<SceneInspector, &SceneInspector::SelectionChanged>(this);
	}

	void SceneInspector::Open()
	{
		EditorWindow* window = GetWindow(SceneInspector::Type);
		window->SetTitle("Inspector");
		window->Show();
	}

	void SceneInspector::OnDrawUI()
	{
		if (m_IsInvalidSelection)
		{
			ImGui::Text("Can't select different object types.");
		}
		else
		{
			if (m_Editor != nullptr)
			{
				m_Editor->DrawInspector();
			}
		}
	}

	void SceneInspector::SelectionChanged()
	{
		if (m_Editor != nullptr)
		{
			ObjectEditor::ReleaseEditor(m_Editor);
		}
		List<Object*> selectedObjects = Selection::GetActiveObjects();
		if (selectedObjects.size() > 0)
		{
			size_t type = selectedObjects[0]->GetType();
			for (size_t i = 1; i < selectedObjects.size(); ++i)
			{
				if (selectedObjects[i]->GetType() != type)
				{
					m_Editor = nullptr;
					m_IsInvalidSelection = true;
					return;
				}
			}
			m_Editor = ObjectEditor::GetEditor(selectedObjects);
			m_IsInvalidSelection = false;
		}
		else
		{
			m_Editor = nullptr;
		}
	}
}