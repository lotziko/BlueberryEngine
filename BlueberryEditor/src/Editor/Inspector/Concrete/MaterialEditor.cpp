#include "MaterialEditor.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Editor\Preview\MaterialPreview.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void MaterialEditor::OnEnable()
	{
		if (s_RenderTexture == nullptr)
		{
			s_RenderTexture = GfxRenderTexturePool::Get(512, 512, 1);
		}

		for (Object* object : m_SerializedObject->GetTargets())
		{
			Material* material = static_cast<Material*>(object);
			Shader* shader = material->GetShader();
			if (shader != nullptr && shader->GetState() == ObjectState::Default)
			{
				auto& data = shader->GetData();
				auto& textureDatas = material->GetTextureDatas();

				for (auto const& propertyData : data.GetProperties())
				{
					if (propertyData.GetType() == PropertyData::PropertyType::Texture)
					{
						TextureData* textureProperty = nullptr;
						auto& textureParameterName = propertyData.GetName();
						for (auto& textureData : textureDatas)
						{
							if (textureData.GetName() == textureParameterName)
							{
								textureProperty = &textureData;
								break;
							}
						}
						if (textureProperty == nullptr)
						{
							TextureData newTextureProperty = {};
							newTextureProperty.SetName(propertyData.GetName());
							material->AddTextureData(newTextureProperty);
						}
					}
				}
			}
		}

		m_ShaderProperty = m_SerializedObject->FindProperty("m_Shader");
		m_TexturesProperty = m_SerializedObject->FindProperty("m_Textures");
	}

	void MaterialEditor::OnDrawInspector()
	{
		ImGui::Property(&m_ShaderProperty, "Shader");

		for (uint32_t i = 0; i < m_TexturesProperty.GetArraySize(); ++i)
		{
			SerializedProperty textureDataProperty = m_TexturesProperty.GetArrayElement(i);
			SerializedProperty nameProperty = textureDataProperty.FindProperty("m_Name");
			SerializedProperty textureProperty = textureDataProperty.FindProperty("m_Texture");
			ImGui::Property(&textureProperty, nameProperty.GetString().c_str());
		}

		if (m_SerializedObject->ApplyModifiedProperties())
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				Material* material = static_cast<Material*>(object);
				material->ApplyProperties();
			}
		}

		if (ImGui::Button("Save"))
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				AssetDB::SetDirty(object);
				ThumbnailCache::Refresh(object);
			}
			AssetDB::SaveAssets();
		}

		static MaterialPreview preview;
		preview.Draw(static_cast<Material*>(m_SerializedObject->GetTarget()), s_RenderTexture);

		ImVec2 size = ImGui::GetContentRegionAvail();
		ImGui::Image(reinterpret_cast<ImTextureID>(s_RenderTexture->GetHandle()), ImVec2(size.x, (s_RenderTexture->GetHeight() * size.x) / static_cast<float>(s_RenderTexture->GetWidth())), ImVec2(0, 1), ImVec2(1, 0));
	}
}