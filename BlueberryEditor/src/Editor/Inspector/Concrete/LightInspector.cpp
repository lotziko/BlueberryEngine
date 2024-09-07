#include "bbpch.h"
#include "LightInspector.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

#include "Editor\Gizmos\Gizmos.h"

namespace Blueberry
{
	const char* LightInspector::GetIconPath(Object* object)
	{
		return "assets/icons/PointLightIcon.png";
	}

	void LightInspector::DrawScene(Object* object)
	{
		auto light = static_cast<Light*>(object);
		auto transform = light->GetEntity()->GetTransform();
		Gizmos::SetMatrix(transform->GetLocalToWorldMatrix());
		Gizmos::DrawCircle(Vector3::Zero, light->GetRange());
	}
}
