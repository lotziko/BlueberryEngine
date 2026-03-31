#include "ModelImporterEditor.h"

#include "Blueberry\Graphics\Material.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void ModelImporterEditor::OnEnable()
	{
		m_MaterialsProperty = m_SerializedObject->FindProperty("m_Materials");
		m_AnimationClipsProperty = m_SerializedObject->FindProperty("m_AnimationClips");
		m_ScaleProperty = m_SerializedObject->FindProperty("m_Scale");
		m_GenerateLightmapUVProperty = m_SerializedObject->FindProperty("m_GenerateLightmapUV");
		m_GeneratePhysicsShapeProperty = m_SerializedObject->FindProperty("m_GeneratePhysicsShape");
	}

	void ModelImporterEditor::OnDrawInspector()
	{
		for (uint32_t i = 0; i < m_MaterialsProperty.GetListSize(); ++i)
		{
			SerializedProperty materialDataProperty = m_MaterialsProperty.GetListElement(i);
			SerializedProperty nameProperty = materialDataProperty.FindProperty("m_Name");
			SerializedProperty materialProperty = materialDataProperty.FindProperty("m_Material");
			ImGui::Property(&materialProperty, nameProperty.GetString().c_str());
		}

		for (uint32_t i = 0; i < m_AnimationClipsProperty.GetListSize(); ++i)
		{
			SerializedProperty animationClipDataProperty = m_AnimationClipsProperty.GetListElement(i);
			SerializedProperty nameProperty = animationClipDataProperty.FindProperty("m_Name");
			SerializedProperty replaceNameProperty = animationClipDataProperty.FindProperty("m_ReplaceName");
			SerializedProperty firstFrameProperty = animationClipDataProperty.FindProperty("m_FirstFrame");
			SerializedProperty lastFrameProperty = animationClipDataProperty.FindProperty("m_LastFrame");
			SerializedProperty isLoopProperty = animationClipDataProperty.FindProperty("m_IsLoop");
			ImGui::Text(nameProperty.GetString().c_str());
			ImGui::Property(&replaceNameProperty, "Replace name");
			ImGui::Property(&firstFrameProperty, "First frame");
			ImGui::Property(&lastFrameProperty, "Last frame");
			ImGui::Property(&isLoopProperty, "Is loop");
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
				ThumbnailCache::Refresh(object);
				ModelImporter* modelImporter = static_cast<ModelImporter*>(object);
				Guid guid = modelImporter->GetGuid();
				for (auto& pair : modelImporter->GetAssetObjects())
				{
					ThumbnailCache::Refresh(ObjectDB::GetObjectFromGuid(guid, pair.first));
				}
			}
			AssetDB::SaveAssets();
			AssetDB::Refresh();
		}
	}
}
