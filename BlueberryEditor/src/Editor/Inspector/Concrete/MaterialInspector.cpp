#include "bbpch.h"
#include "MaterialInspector.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "imgui\imgui.h"

#include "Editor\Panels\Scene\SceneArea.h"

namespace Blueberry
{
	void MaterialInspector::Draw(Object* object)
	{
		Material* material = static_cast<Material*>(object);
		Shader* shader = material->GetShader();
		if (ImGui::ObjectEdit("Shader", (Object**)&shader, Shader::Type))
		{
			material->SetShader(shader);
		}

		if (shader != nullptr && (shader->GetState() == ObjectState::Default))
		{
			auto& data = shader->GetData();
			auto& textureDatas = material->GetTextureDatas();
			bool hasPropertyChanges = false;

			for (auto const& propertyData : data.GetProperties())
			{
				if (propertyData.GetType() == PropertyData::PropertyType::Texture)
				{
					TextureData* textureProperty = nullptr;
					auto& textureParameterName = propertyData.GetName();
					Texture* texture = nullptr;
					for (auto& textureData : textureDatas)
					{
						if (textureData.GetName() == textureParameterName)
						{
							texture = textureData.GetTexture();
							textureProperty = &textureData;
						}
					}

					if (textureProperty != nullptr)
					{
						if (ImGui::ObjectEdit(propertyData.GetName().c_str(), (Object**)&texture, propertyData.GetTextureDimension() == TextureDimension::TextureCube ? TextureCube::Type : Texture2D::Type))
						{
							textureProperty->SetTexture(texture);
							hasPropertyChanges = true;
						}
					}
					else
					{
						TextureData newTextureProperty = {};
						newTextureProperty.SetName(propertyData.GetName());
						material->AddTextureData(newTextureProperty);
					}
				}
			}

			if (hasPropertyChanges)
			{
				material->ApplyProperties();
				ThumbnailCache::Refresh(material);
				SceneArea::RequestRedrawAll();
			}
		}

		if (ImGui::Button("Save"))
		{
			AssetDB::SetDirty(material);
			AssetDB::SaveAssets();
		}
	}
}