#include "LightEditor.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture2D.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static Texture* s_SpotIcon = nullptr;
	static Texture* s_PointIcon = nullptr;

	LightEditor::LightEditor()
	{
		if (s_SpotIcon == nullptr)
		{
			s_SpotIcon = static_cast<Texture*>(AssetLoader::Load("assets/icons/SpotLightIcon.png"));
			s_PointIcon = static_cast<Texture*>(AssetLoader::Load("assets/icons/PointLightIcon.png"));
		}
	}

	void LightEditor::OnEnable()
	{
		m_TypeProperty = m_SerializedObject->FindProperty("m_Type");
		m_ColorProperty = m_SerializedObject->FindProperty("m_Color");
		m_IntensityProperty = m_SerializedObject->FindProperty("m_Intensity");
		m_RangeProperty = m_SerializedObject->FindProperty("m_Range");
		m_OuterSpotAngleProperty = m_SerializedObject->FindProperty("m_OuterSpotAngle");
		m_InnerSpotAngleProperty = m_SerializedObject->FindProperty("m_InnerSpotAngle");
		m_IsCastingShadowsProperty = m_SerializedObject->FindProperty("m_IsCastingShadows");
		m_IsCastingFogProperty = m_SerializedObject->FindProperty("m_IsCastingFog");
		m_IsCachedProperty = m_SerializedObject->FindProperty("m_IsCached");
		m_CookieProperty = m_SerializedObject->FindProperty("m_Cookie");
	}

	void LightEditor::OnDrawInspector()
	{
		if (m_SerializedObject->GetTargets().size() == 1)
		{
			ImGui::Property(&m_TypeProperty, "Type");
			ImGui::Property(&m_ColorProperty, "Color");
			ImGui::Property(&m_IntensityProperty, "Intensity");

			LightType type = m_TypeProperty.GetEnum<LightType>();
			if (type != LightType::Directional)
			{
				ImGui::Property(&m_RangeProperty, "Range");
				if (type == LightType::Spot)
				{
					ImGui::Property(&m_OuterSpotAngleProperty, "Outer Angle");
					ImGui::Property(&m_InnerSpotAngleProperty, "Inner Angle");
				}
			}

			ImGui::Property(&m_IsCastingShadowsProperty, "Is Casting Shadows");
			ImGui::Property(&m_IsCastingFogProperty, "Is Casting Fog");
			ImGui::Property(&m_IsCachedProperty, "Is Cached");
			
			if (type == LightType::Spot)
			{
				ImGui::Property(&m_CookieProperty, "Cookie");
			}

			m_SerializedObject->ApplyModifiedProperties();
		}
	}

	Texture* LightEditor::GetIcon(Object* object)
	{
		Light* light = static_cast<Light*>(object);
		switch (light->GetType())
		{
		case LightType::Spot:
			return s_SpotIcon;
		case LightType::Point:
			return s_PointIcon;
		default:
			return nullptr;
		}
	}

	void LightEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			Light* light = static_cast<Light*>(target);
			Transform* transform = light->GetTransform();
			LightType type = light->GetType();
			float range = light->GetRange();

			if (type == LightType::Point)
			{
				Gizmos::SetMatrix(Math::CreateTRS(transform->GetPosition(), Quaternion::Identity, Vector3::One));
				Gizmos::DrawSphere(Vector3::Zero, range);
			}
			else if (type == LightType::Spot)
			{
				Gizmos::SetMatrix(Math::CreateTRS(transform->GetPosition(), transform->GetRotation(), Vector3::One));

				float outerAngle = light->GetOuterSpotAngle();
				float innerAngle = light->GetInnerSpotAngle();
				float radianHalfOuterAngle = Math::Math::ToRadians(outerAngle) * 0.5f;
				float radianHalfInnerAngle = Math::Math::ToRadians(innerAngle) * 0.5f;

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
			else if (type == LightType::Directional)
			{
				float radius = 0.5f;
				Gizmos::SetMatrix(Math::CreateTRS(transform->GetPosition(), transform->GetRotation(), Vector3::One));
				Gizmos::SetColor(Color(1, 1, 1, 1));
				Gizmos::DrawDisc(Vector3::Zero, Vector3::Forward, radius);

				uint32_t lineCount = 8;
				for (uint32_t i = 0; i < lineCount; ++i)
				{
					float angle = 2 * Math::Pi * static_cast<float>(i) / lineCount;
					Vector3 position = Vector3(std::cos(angle) * radius, std::sin(angle) * radius, 0);
					Gizmos::DrawLine(position, position + Vector3(0.0f, 0.0f, 0.75f));
				}
			}
		}
	}

	void LightEditor::DrawCone(const float& radius, const float& height, const int& mask)
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
