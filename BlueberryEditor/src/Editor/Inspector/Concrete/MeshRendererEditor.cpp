#include "MeshRendererEditor.h"

#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Editor\Gizmos\Gizmos.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	void MeshRendererEditor::OnEnable()
	{
		m_MeshProperty = m_SerializedObject->FindProperty("m_Mesh");
		m_MaterialsProperty = m_SerializedObject->FindProperty("m_Materials");
		m_IsCastingShadowsProperty = m_SerializedObject->FindProperty("m_IsCastingShadows");
		m_IsBakeableProperty = m_SerializedObject->FindProperty("m_IsBakeable");
	}

	void MeshRendererEditor::OnDrawInspector()
	{
		ImGui::Property(&m_MeshProperty);
		ImGui::Property(&m_MaterialsProperty);
		ImGui::Property(&m_IsCastingShadowsProperty);
		ImGui::Property(&m_IsBakeableProperty);
		m_SerializedObject->ApplyModifiedProperties();
	}

	void MeshRendererEditor::OnDrawSceneSelected()
	{
		/*for (Object* target : m_SerializedObject->GetTargets())
		{
			MeshRenderer* renderer = static_cast<MeshRenderer*>(target);
			AABB bounds = renderer->GetBounds();
			Gizmos::SetMatrix(Matrix::Identity);
			Gizmos::DrawBox(bounds.Center, static_cast<Vector3>(bounds.Extents) * 2);
		}*/
	}
}
