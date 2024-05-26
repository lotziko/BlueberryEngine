#include "bbpch.h"
#include "MaterialInspector.h"

#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "imgui\imgui.h"

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

		if (shader != nullptr)
		{
			auto data = shader->GetData();
			auto textureDatas = material->GetTextureDatas();

			for (auto const& textureParameter : data->GetTextureParameters())
			{
				TextureData* textureProperty = nullptr;
				auto& textureParameterName = textureParameter.Get()->GetName();
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
					if (ImGui::ObjectEdit(textureParameter.Get()->GetName().c_str(), (Object**)&texture, Texture::Type))
					{
						textureProperty->SetTexture(texture);
					}
				}
				else
				{
					textureProperty = new TextureData();
					textureProperty->SetName(textureParameter.Get()->GetName());
					material->AddTextureData(textureProperty);
				}
			}
		}

		if (ImGui::Button("Save"))
		{
			AssetDB::SetDirty(material);
			AssetDB::SaveAssets();
		}
	}
}