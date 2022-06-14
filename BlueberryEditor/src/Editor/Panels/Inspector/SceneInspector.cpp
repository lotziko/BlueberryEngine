#include "bbpch.h"
#include "SceneInspector.h"

#include "Blueberry\Scene\Scene.h"

#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Selection.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void SceneInspector::DrawUI()
	{
		ImGui::Begin("Inspector");

		Object* selectedObject = Selection::GetActiveObject();
		if (selectedObject != nullptr)
		{
			std::size_t type = selectedObject->GetType();
			ObjectInspector* inspector = ObjectInspectors::GetInspector(type);
			if (inspector != nullptr)
			{
				inspector->Draw(selectedObject);
			}
		}

		ImGui::End();
	}
}