#include "bbpch.h"
#include "TransformInspector.h"

#include "Blueberry\Scene\Components\Transform.h"

#include "imgui\imgui.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	OBJECT_INSPECTOR_DECLARATION(TransformInspector, Transform)

	void TransformInspector::Draw(Object* object)
	{
		Transform* transform = static_cast<Transform*>(object);

		Vector3 localPosition = transform->GetLocalPosition();
		if (ImGui::DragVector3("Position", localPosition))
		{
			transform->SetLocalPosition(localPosition);
		}

		Vector3 localRotation = ToDegrees(transform->GetLocalEulerRotation());
		std::intptr_t transformId = reinterpret_cast<std::intptr_t>(transform);
		if (m_TransformEulerCache.count(transformId) > 0)
		{
			localRotation = m_TransformEulerCache[transformId];
		}

		if (ImGui::DragVector3("Rotation", localRotation))
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
		}

		Vector3 localScale = transform->GetLocalScale();
		if (ImGui::DragVector3("Scale", localScale))
		{
			transform->SetLocalScale(localScale);
		}
	}
}