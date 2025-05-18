#include "SceneInspector.h"

#include "InspectorExpandedItemsCache.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Selection.h"
#include "Editor\Menu\EditorMenuManager.h"

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
	}

	SceneInspector::~SceneInspector()
	{
		InspectorExpandedItemsCache::Save();
	}

	void SceneInspector::Open()
	{
		EditorWindow* window = GetWindow(SceneInspector::Type);
		window->SetTitle("Inspector");
		window->Show();
	}

	void SceneInspector::OnDrawUI()
	{
		Object* selectedObject = Selection::GetActiveObject();
		if (selectedObject != nullptr)
		{
			size_t type = selectedObject->GetType();
			ObjectInspector* inspector = ObjectInspectorDB::GetInspector(type);
			if (inspector != nullptr)
			{
				inspector->Draw(selectedObject);
			}
		}
	}
}