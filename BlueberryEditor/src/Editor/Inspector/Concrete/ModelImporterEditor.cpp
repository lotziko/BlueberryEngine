#include "ModelImporterEditor.h"

#include "Blueberry\Graphics\Material.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void ModelImporterEditor::OnEnable()
	{
		m_MaterialsProperty = m_SerializedObject->FindProperty("m_Materials");
		m_ScaleProperty = m_SerializedObject->FindProperty("m_Scale");
		m_GenerateLightmapUVProperty = m_SerializedObject->FindProperty("m_GenerateLightmapUV");
		m_GeneratePhysicsShapeProperty = m_SerializedObject->FindProperty("m_GeneratePhysicsShape");
	}

	void ModelImporterEditor::OnDrawInspector()
	{
		for (uint32_t i = 0; i < m_MaterialsProperty.GetArraySize(); ++i)
		{
			SerializedProperty materialDataProperty = m_MaterialsProperty.GetArrayElement(i);
			SerializedProperty nameProperty = materialDataProperty.FindProperty("m_Name");
			SerializedProperty materialProperty = materialDataProperty.FindProperty("m_Material");
			ImGui::Property(&materialProperty, nameProperty.GetString().c_str());
		}

		ImGui::Property(&m_ScaleProperty, "Scale");
		ImGui::Property(&m_GenerateLightmapUVProperty, "Generate Lightmap UV");
		ImGui::Property(&m_GeneratePhysicsShapeProperty, "Generate Physics Shape");
		m_SerializedObject->ApplyModifiedProperties();

		if (ImGui::Button("Save"))
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				AssetDB::SetDirty(object);
			}
			AssetDB::SaveAssets();
		}
	}
}
