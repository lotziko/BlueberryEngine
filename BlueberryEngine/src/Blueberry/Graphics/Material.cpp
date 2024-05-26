#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	DATA_DEFINITION(TextureData)
	OBJECT_DEFINITION(Object, Material)

	const std::string& TextureData::GetName()
	{
		return m_Name;
	}

	void TextureData::SetName(const std::string& name)
	{
		m_Name = name;
	}

	Texture* TextureData::GetTexture()
	{
		return m_Texture.Get();
	}

	void TextureData::SetTexture(Texture* texture)
	{
		m_Texture = texture;
	}

	void TextureData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(TextureData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &TextureData::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_Texture), &TextureData::m_Texture, BindingType::ObjectPtr).SetObjectType(Texture::Type))
		END_OBJECT_BINDING()
	}

	Material* Material::Create(Shader* shader)
	{
		Material* material = Object::Create<Material>();
		material->SetShader(shader);
		material->ApplyShaderProperties();
		material->OnCreate();
		return material;
	}

	void Material::SetTexture(std::size_t id, Texture* texture)
	{
		m_GfxTextures.insert_or_assign(id, texture->m_Texture);
	}

	void Material::SetTexture(std::string name, Texture* texture)
	{
		SetTexture(TO_HASH(name), texture);
	}

	Shader* Material::GetShader()
	{
		return m_Shader.Get();
	}

	void Material::SetShader(Shader* shader)
	{
		m_Shader = shader;
	}

	void Material::ApplyShaderProperties()
	{
		/*if (m_Shader.IsValid())
		{
			for (auto& textureParameter : m_Shader->GetData()->GetTextureParameters())
			{
				m_Textures.
			}
		}*/
	}

	const ShaderData* Material::GetShaderData()
	{
		if (m_Shader.IsValid())
		{
			return m_Shader.Get()->GetData();
		}
		return nullptr;
	}

	std::vector<DataPtr<TextureData>>& Material::GetTextureDatas()
	{
		return m_Textures;
	}

	void Material::AddTextureData(TextureData* data)
	{
		m_Textures.emplace_back(DataPtr<TextureData>(data));
		FillGfxTextures();
	}

	void Material::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Material)
		BIND_FIELD(FieldInfo(TO_STRING(m_Shader), &Material::m_Shader, BindingType::ObjectPtr).SetObjectType(Shader::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Textures), &Material::m_Textures, BindingType::DataArray).SetObjectType(TextureData::Type))
		END_OBJECT_BINDING()
	}

	void Material::OnCreate()
	{
		FillGfxTextures();
	}

	void Material::FillGfxTextures()
	{
		m_GfxTextures.clear();
		for (auto const& texture : m_Textures)
		{
			if (texture.Get()->m_Texture.IsValid())
			{
				m_GfxTextures.insert_or_assign(TO_HASH(texture.Get()->m_Name), texture.Get()->m_Texture->m_Texture);
			}
		}
	}
}