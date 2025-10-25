#include "Blueberry\Graphics\Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Tools\CRCHelper.h"

namespace Blueberry
{
	DATA_DEFINITION(TextureData)
	{
		DEFINE_FIELD(TextureData, m_Name, BindingType::String, {})
		DEFINE_FIELD(TextureData, m_Texture, BindingType::ObjectPtr, FieldOptions().SetObjectType(Texture::Type))
	}

	OBJECT_DEFINITION(Material, Object)
	{
		DEFINE_BASE_FIELDS(Material, Object)
		DEFINE_FIELD(Material, m_Shader, BindingType::ObjectPtr, FieldOptions().SetObjectType(Shader::Type))
		DEFINE_FIELD(Material, m_Textures, BindingType::DataList, FieldOptions().SetObjectType(TextureData::Type))
		DEFINE_FIELD(Material, m_ActiveKeywords, BindingType::StringList, {})
	}

	const String& TextureData::GetName()
	{
		return m_Name;
	}

	void TextureData::SetName(const String& name)
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

	Material* Material::Create(Shader* shader)
	{
		Material* material = Object::Create<Material>();
		material->SetShader(shader);
		material->OnCreate();
		return material;
	}

	void Material::SetTexture(size_t id, Texture* texture)
	{
		ApplyTextureBinding(id, texture);
		m_Crc = UINT32_MAX;
	}

	void Material::SetTexture(String name, Texture* texture)
	{
		SetTexture(TO_HASH(name), texture);
	}

	Shader* Material::GetShader()
	{
		if (m_BindedTextures.size() < m_Textures.size())
		{
			ApplyProperties();
		}

		return m_Shader.Get();
	}

	void Material::SetShader(Shader* shader)
	{
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			m_Shader->m_Dependencies.erase(m_ObjectId);
		}
		m_Shader = shader;
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			m_Shader->m_Dependencies.emplace(m_ObjectId);
		}
	}

	void Material::ApplyProperties()
	{
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			m_Shader->m_Dependencies.emplace(m_ObjectId);
		}
		FillTextureMap();
		m_Crc = UINT32_MAX;
	}

	const ShaderData* Material::GetShaderData()
	{
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			return &m_Shader.Get()->GetData();
		}
		return nullptr;
	}

	List<TextureData>& Material::GetTextureDatas()
	{
		return m_Textures;
	}

	void Material::AddTextureData(const TextureData& data)
	{
		m_Textures.emplace_back(data);
		FillTextureMap();
	}

	void Material::SetKeyword(const String& keyword, const bool& enabled)
	{
		// TODO use an unordered_map
		auto it = std::find(m_ActiveKeywords.begin(), m_ActiveKeywords.end(), keyword);
		if (it != m_ActiveKeywords.end() && !enabled)
		{
			m_ActiveKeywords.erase(it);
		}
		else if (enabled)
		{
			m_ActiveKeywords.emplace_back(keyword);
		}
		m_ActiveKeywordsMask = 0;
		for (auto keyword : m_ActiveKeywords)
		{
			m_ActiveKeywordsMask |= m_Shader->m_LocalKeywords.GetMask(TO_HASH(keyword));
		}
	}

	const uint32_t& Material::GetActiveKeywordsMask()
	{
		return m_ActiveKeywordsMask;
	}

	const uint32_t& Material::GetCRC()
	{
		if (m_Crc == UINT32_MAX)
		{
			m_Crc = 0;
			m_Crc = CRCHelper::Calculate(m_Shader->m_ObjectId, m_Crc);
			m_Crc = CRCHelper::Calculate(m_Shader->m_UpdateCount, m_Crc);
			for (auto& binding : m_BindedTextures)
			{
				Texture* texture = static_cast<Texture*>(ObjectDB::GetObject(binding.objectId));
				m_Crc = CRCHelper::Calculate(&binding, sizeof(TextureBinding), m_Crc);
				m_Crc = CRCHelper::Calculate(texture->m_UpdateCount, m_Crc);
			}
			for (auto& keyword : m_ActiveKeywords)
			{
				m_Crc = CRCHelper::Calculate(&keyword, keyword.size(), m_Crc);
			}
		}
		return m_Crc;
	}

	Texture* Material::GetTexture(const size_t& id)
	{
		if (m_BindedTextures.size() < m_Textures.size())
		{
			ApplyProperties();
		}

		for (auto& binding : m_BindedTextures)
		{
			if (binding.id == id)
			{
				return static_cast<Texture*>(ObjectDB::GetObject(binding.objectId));
			}
		}
		return nullptr;
	}

	void Material::OnNotify()
	{
		m_Crc = UINT32_MAX;
	}

	void Material::FillTextureMap()
	{
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			for (auto& property : m_Shader->GetData().GetProperties())
			{
				if (property.GetType() == PropertyData::PropertyType::Texture)
				{
					size_t key = TO_HASH(property.GetName());
					Texture* texture = DefaultTextures::GetTexture(property.GetDefaultTextureName(), property.GetTextureDimension());
					if (texture != nullptr)
					{
						ApplyTextureBinding(key, texture);
					}
				}
			}
		}
		for (auto const& texture : m_Textures)
		{
			if (texture.m_Texture.IsValid())
			{
				size_t key = TO_HASH(texture.m_Name);
				Texture* tex = texture.m_Texture.Get();
				ApplyTextureBinding(key, tex);
				if (tex->GetState() == ObjectState::Default)
				{
					tex->m_Dependencies.emplace(m_ObjectId);
				}
			}
		}
	}

	void Material::ApplyTextureBinding(const size_t& id, Texture* texture)
	{
		for (size_t i = 0; i < m_BindedTextures.size(); ++i)
		{
			auto& binding = m_BindedTextures[i];
			if (binding.id == id)
			{
				binding.objectId = texture->GetObjectId();
				binding.index = texture->Get()->m_Index;
				return;
			}
		}
		m_BindedTextures.push_back({ id, texture->GetObjectId(), texture->Get()->m_Index });
	}
}