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
		Light* light = static_cast<Light*>(object);
		switch (light->GetType())
		{
		case LightType::Spot:
			return "assets/icons/SpotLightIcon.png";
		case LightType::Point:
			return "assets/icons/PointLightIcon.png";
		default:
			return nullptr;
		}
	}

	void LightInspector::DrawScene(Object* object)
	{
		Light* light = static_cast<Light*>(object);
		Transform* transform = light->GetTransform();
		LightType type = light->GetType();
		float range = light->GetRange();

		if (type == LightType::Point)
		{
			Gizmos::SetMatrix(CreateTRS(transform->GetPosition(), Quaternion::Identity, Vector3::One));
			Gizmos::DrawSphere(Vector3::Zero, range);
		}
		else if (type == LightType::Spot)
		{
			Gizmos::SetMatrix(CreateTRS(transform->GetPosition(), transform->GetRotation(), Vector3::One));

			float outerAngle = light->GetOuterSpotAngle();
			float innerAngle = light->GetInnerSpotAngle();
			float radianHalfOuterAngle = ToRadians(outerAngle) * 0.5f;
			float radianHalfInnerAngle = ToRadians(innerAngle) * 0.5f;

			float outerDiscRadius = range * sin(radianHalfOuterAngle);
			float outerDiscDistance = range * cos(radianHalfOuterAngle);

			Vector3 vectorLineUp = Vector3::UnitZ * outerDiscDistance + Vector3::UnitY * outerDiscRadius;
			vectorLineUp.Normalize();
			Vector3 vectorLineLeft = Vector3::UnitZ * outerDiscDistance - Vector3::UnitX * outerDiscRadius;
			vectorLineLeft.Normalize();

			if (innerAngle > 0.0f)
			{
				float innerDiscRadius = range * sin(radianHalfInnerAngle);
				float innerDiscDistance = range * cos(radianHalfInnerAngle);

				Gizmos::SetColor(Color(1, 1, 1, 0.4f));
				DrawCone(innerDiscRadius, innerDiscDistance, 1 | 2);
			}

			Gizmos::SetColor(Color(1, 1, 1, 0.4f));
			Vector3 rangeCenter = Vector3::UnitZ * range;
			Gizmos::DrawLine(Vector3::Zero, rangeCenter);

			Gizmos::SetColor(Color(1, 1, 1, 1));
			Gizmos::DrawArc(Vector3::Zero, Vector3::UnitX, vectorLineUp, outerAngle, range);
			Gizmos::DrawArc(Vector3::Zero, Vector3::UnitY, vectorLineLeft, outerAngle, range);

			DrawCone(outerDiscRadius, outerDiscDistance, 4 | 8);
		}
	}

	void LightInspector::DrawCone(const float& radius, const float& height, const int& mask)
	{
		Vector3 rangeCenter = Vector3::UnitZ * height;

		if (mask & 1)
		{
			Gizmos::DrawLine(Vector3::Zero, rangeCenter + Vector3::UnitY * radius);
		}
		if (mask & 2)
		{
			Gizmos::DrawLine(Vector3::Zero, rangeCenter - Vector3::UnitY * radius);
		}
		if (mask & 4)
		{
			Gizmos::DrawLine(Vector3::Zero, rangeCenter + Vector3::UnitX * radius);
		}
		if (mask & 8)
		{
			Gizmos::DrawLine(Vector3::Zero, rangeCenter - Vector3::UnitX * radius);
		}
		
		Gizmos::DrawDisc(rangeCenter, Vector3::UnitZ, radius);
	}
}
