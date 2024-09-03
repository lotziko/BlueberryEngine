#include "bbpch.h"
#include "TransformInspector.h"

#include "Blueberry\Scene\Components\Transform.h"

#include "imgui\imgui.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Editor\Panels\Scene\SceneArea.h"

namespace Blueberry
{
	void TransformInspector::Draw(Object* object)
	{
		Transform* transform = static_cast<Transform*>(object);
		bool hasChange = false;

		Vector3 localPosition = transform->GetLocalPosition();
		if (ImGui::DragVector3("Position", &localPosition))
		{
			transform->SetLocalPosition(localPosition);
			hasChange = true;
		}

		Vector3 localRotation = ToDegrees(transform->GetLocalEulerRotation());
		std::intptr_t transformId = reinterpret_cast<std::intptr_t>(transform);
		if (m_TransformEulerCache.count(transformId) > 0)
		{
			localRotation = m_TransformEulerCache[transformId];
		}

		if (ImGui::DragVector3("Rotation", &localRotation))
		{
			transform->SetLocalEulerRotation(ToRadians(localRotation));
			if (ImGui::IsItemActive())
			{
				m_TransformEulerCache[transformId] = localRotation;
			}
			else
			{
				m_TransformEulerCache.erase(transformId);
			}
			hasChange = true;
		}

		Vector3 localScale = transform->GetLocalScale();
		if (ImGui::DragVector3("Scale", &localScale))
		{
			transform->SetLocalScale(localScale);
			hasChange = true;
		}

		if (hasChange)
		{
			SceneArea::RequestRedrawAll();
		}
	}
}