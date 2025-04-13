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
			auto data = shader->GetData();
			auto textureDatas = material->GetTextureDatas();
			bool hasPropertyChanges = false;

			for (auto const& propertyData : data->GetProperties())
			{
				if (propertyData.Get()->GetType() == PropertyData::PropertyType::Texture)
				{
					TextureData* textureProperty = nullptr;
					auto& textureParameterName = propertyData.Get()->GetName();
					Texture* texture = nullptr;
					for (auto const& textureData : textureDatas)
					{
						if (textureData.Get()->GetName() == textureParameterName)
						{
							texture = textureData.Get()->GetTexture();
							textureProperty = textureData.Get();
						}
					}

					if (textureProperty != nullptr)
					{
						if (ImGui::ObjectEdit(propertyData.Get()->GetName().c_str(), (Object**)&texture, propertyData.Get()->GetTextureDimension() == TextureDimension::TextureCube ? TextureCube::Type : Texture2D::Type))
						{
							textureProperty->SetTexture(texture);
							hasPropertyChanges = true;
						}
					}
					else
					{
						textureProperty = new TextureData();
						textureProperty->SetName(propertyData.Get()->GetName());
						material->AddTextureData(textureProperty);
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