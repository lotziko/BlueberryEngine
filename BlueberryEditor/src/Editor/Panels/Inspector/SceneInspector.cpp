#include "bbpch.h"
#include "SceneInspector.h"

#include "InspectorExpandedItemsCache.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Selection.h"
#include "Editor\Menu\EditorMenuManager.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	OBJECT_DEFINITION(EditorWindow, SceneInspector)

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

	void SceneInspector::BindProperties()
	{
		BEGIN_OBJECT_BINDING(SceneInspector)
		BIND_FIELD(FieldInfo(TO_STRING(m_Title), &SceneInspector::m_Title, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &SceneInspector::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()

		EditorMenuManager::AddItem("Window/Inspector", &SceneInspector::Open);
	}

	void SceneInspector::OnDrawUI()
	{
		Object* selectedObject = Selection::GetActiveObject();
		if (selectedObject != nullptr)
		{
			std::size_t type = selectedObject->GetType();
			ObjectInspector* inspector = ObjectInspectorDB::GetInspector(type);
			if (inspector != nullptr)
			{
				inspector->Draw(selectedObject);
			}
		}
	}
}