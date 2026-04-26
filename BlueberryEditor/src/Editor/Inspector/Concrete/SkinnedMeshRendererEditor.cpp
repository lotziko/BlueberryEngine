#include "SkinnedMeshRendererEditor.h"

#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	void SkinnedMeshRendererEditor::OnEnable()
	{
		m_MeshProperty = m_SerializedObject->FindProperty("m_Mesh");
		m_RootProperty = m_SerializedObject->FindProperty("m_Root");
		m_MaterialsProperty = m_SerializedObject->FindProperty("m_Materials");
		m_IsCastingShadowsProperty = m_SerializedObject->FindProperty("m_IsCastingShadows");
	}

	void SkinnedMeshRendererEditor::OnDrawInspector()
	{
		ImGui::Property(&m_MeshProperty);
		ImGui::Property(&m_RootProperty);
		ImGui::Property(&m_MaterialsProperty);
		ImGui::Property(&m_IsCastingShadowsProperty);
		m_SerializedObject->ApplyModifiedProperties();
	}

	void SkinnedMeshRendererEditor::OnDrawSceneSelected()
	{
		for (Object* target : m_SerializedObject->GetTargets())
		{
			SkinnedMeshRenderer* renderer = static_cast<SkinnedMeshRenderer*>(target);
			AABB bounds = renderer->GetBounds();
			Gizmos::SetMatrix(Matrix::Identity);
			Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
		}
	}
}
