#include "ModelImporterEditor.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Scene\Entity.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Preview\EntityPreview.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	GfxTexture* ModelImporterEditor::s_RenderTexture = nullptr;

	void ModelImporterEditor::OnEnable()
	{
		m_MaterialsProperty = m_SerializedObject->FindProperty("m_Materials");
		m_AnimationClipsProperty = m_SerializedObject->FindProperty("m_AnimationClips");
		m_ScaleProperty = m_SerializedObject->FindProperty("m_Scale");
		m_GenerateLightmapUVProperty = m_SerializedObject->FindProperty("m_GenerateLightmapUV");
		m_GeneratePhysicsShapeProperty = m_SerializedObject->FindProperty("m_GeneratePhysicsShape");

		if (s_RenderTexture == nullptr)
		{
			s_RenderTexture = GfxTexturePool::Get(512, 512, 1, TextureUsageFlags::RenderTarget);
		}
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
			}
			AssetDB::SaveAssets();
			AssetDB::Refresh();
		}

		ImVec2 size = ImGui::GetContentRegionAvail();
		ImVec2 imageSize = ImVec2(size.x, (s_RenderTexture->GetHeight() * size.x) / static_cast<float>(s_RenderTexture->GetWidth()));

		if (m_SerializedObject->GetTargets().size() == 1)
		{
			AssetImporter* importer = static_cast<AssetImporter*>(m_SerializedObject->GetTarget());
			Entity* entity = static_cast<Entity*>(ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject()));
			if (entity != nullptr)
			{
				static EntityPreview preview;
				preview.Draw(entity, s_RenderTexture);
				ImGui::Image(reinterpret_cast<ImTextureID>(s_RenderTexture->GetHandle()), imageSize);
			}
		}
	}
}
