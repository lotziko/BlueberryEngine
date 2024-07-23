#include "bbpch.h"
#include "SceneInspector.h"

#include "InspectorExpandedItemsCache.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"
#include "Editor\Selection.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneInspector::SceneInspector()
	{
		InspectorExpandedItemsCache::Load();
	}

	SceneInspector::~SceneInspector()
	{
		InspectorExpandedItemsCache::Save();
	}

	void SceneInspector::DrawUI()
	{
		ImGui::Begin("Inspector");

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

		ImGui::End();
	}
}