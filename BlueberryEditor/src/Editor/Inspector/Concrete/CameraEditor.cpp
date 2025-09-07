#include "CameraEditor.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

#include "Editor\Misc\ImGuiHelper.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Texture2D.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	static Texture* s_Icon = nullptr;

	CameraEditor::CameraEditor()
	{
		if (s_Icon == nullptr)
		{
			s_Icon = static_cast<Texture*>(AssetLoader::Load("assets/icons/CameraIcon.png"));
		}
	}

	void CameraEditor::OnEnable()
	{
		m_IsOrthographicProperty = m_SerializedObject->FindProperty("m_IsOrthographic");
		m_OrthographicSizeProperty = m_SerializedObject->FindProperty("m_OrthographicSize");
		m_PixelSizeProperty = m_SerializedObject->FindProperty("m_PixelSize");
		m_FieldOfViewProperty = m_SerializedObject->FindProperty("m_FieldOfView");
		m_AspectRatioProperty = m_SerializedObject->FindProperty("m_AspectRatio");
		m_ZNearPlaneProperty = m_SerializedObject->FindProperty("m_ZNearPlane");
		m_ZFarPlaneProperty = m_SerializedObject->FindProperty("m_ZFarPlane");
	}

	void CameraEditor::OnDrawInspector()
	{
		ImGui::Property(&m_IsOrthographicProperty, "Is Orthographic");

		if (m_IsOrthographicProperty.GetBool())
		{
			ImGui::Property(&m_OrthographicSizeProperty, "Orthographic size");
			if (ImGui::Property(&m_PixelSizeProperty, "Pixel size"))
			{
				Vector2 pixelSize = m_PixelSizeProperty.GetVector2();
				m_AspectRatioProperty.SetFloat(pixelSize.x / pixelSize.y);
			}
		}
		else
		{
			ImGui::Property(&m_FieldOfViewProperty, "Field of View");
		}

		ImGui::Property(&m_ZNearPlaneProperty, "Near plane");
		ImGui::Property(&m_ZFarPlaneProperty, "Far plane");

		if (m_SerializedObject->ApplyModifiedProperties())
		{
			SceneArea::RequestRedrawAll();
		}
	}

	Texture* CameraEditor::GetIcon(Object* object)
	{
		return s_Icon;
	}

	void CameraEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			Camera* camera = static_cast<Camera*>(target);
			Transform* transform = camera->GetTransform();

			Matrix view = camera->GetInverseViewMatrix();
			Matrix projection = camera->GetProjectionMatrix();
			Frustum frustum;
			frustum.CreateFromMatrix(frustum, projection, false);
			frustum.Transform(frustum, view);

			Gizmos::SetMatrix(Matrix::Identity);
			Gizmos::DrawFrustum(frustum);
		}
	}
}