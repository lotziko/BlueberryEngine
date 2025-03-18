#include "bbpch.h"
#include "Material.h"

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
		material->OnCreate();
		return material;
	}

	void Material::SetTexture(size_t id, Texture* texture)
	{
		m_BindedTextures[id] = texture->GetObjectId();
		m_Crc = UINT32_MAX;
	}

	void Material::SetTexture(std::string name, Texture* texture)
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
		m_Shader = shader;
	}

	void Material::ApplyProperties()
	{
		FillTextureMap();
		m_Crc = UINT32_MAX;
	}

	const ShaderData* Material::GetShaderData()
	{
		if (m_Shader.IsValid())
		{
			return m_Shader.Get()->GetData();
		}
		return nullptr;
	}

	List<DataPtr<TextureData>>& Material::GetTextureDatas()
	{
		return m_Textures;
	}

	void Material::AddTextureData(TextureData* data)
	{
		m_Textures.emplace_back(DataPtr<TextureData>(data));
		FillTextureMap();
	}

	void Material::SetKeyword(const std::string& keyword, const bool& enabled)
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
			for (auto& pair : m_BindedTextures)
			{
				m_Crc = CRCHelper::Calculate(&pair, sizeof(std::pair<size_t, ObjectId>), m_Crc);
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
		auto it = m_BindedTextures.find(id);
		if (it != m_BindedTextures.end())
		{
			return static_cast<Texture*>(ObjectDB::GetObject(it->second));
		}
		return nullptr;
	}

	void Material::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Material)
		BIND_FIELD(FieldInfo(TO_STRING(m_Shader), &Material::m_Shader, BindingType::ObjectPtr).SetObjectType(Shader::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Textures), &Material::m_Textures, BindingType::DataArray).SetObjectType(TextureData::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_ActiveKeywords), &Material::m_ActiveKeywords, BindingType::StringArray))
		END_OBJECT_BINDING()
	}

	void Material::FillTextureMap()
	{
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			const ShaderData* shaderData = m_Shader->GetData();
			for (auto& parameter : shaderData->GetTextureParameters())
			{
				TextureParameterData* data = parameter.Get();
				size_t key = TO_HASH(data->GetName());
				m_BindedTextures.try_emplace(key, DefaultTextures::GetTexture(data->GetDefaultTextureName())->GetObjectId());
			}
		}
		for (auto const& texture : m_Textures)
		{
			TextureData* textureData = texture.Get();
			if (textureData->m_Texture.IsValid())
			{
				size_t key = TO_HASH(textureData->m_Name);
				m_BindedTextures.insert_or_assign(key, textureData->m_Texture.Get()->GetObjectId());
			}
		}
	}
}