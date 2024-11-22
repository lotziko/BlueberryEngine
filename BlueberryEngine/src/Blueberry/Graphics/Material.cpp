#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
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
		m_TextureMap.insert_or_assign(id, texture->GetObjectId());
		m_Crc = -1;
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

	void Material::ApplyProperties()
	{
		if (m_Shader.IsValid())
		{
			m_PassCache.resize(m_Shader.Get()->GetData()->GetPassCount());
		}
		FillTextureMap();
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
		FillTextureMap();
	}

	void Material::SetKeyword(const std::string& keyword, const bool& enabled)
	{
		// TODO use an unordered_map
		if (enabled)
		{
			m_ActiveKeywords.emplace_back(keyword);
		}
	}

	GfxRenderState* Material::GetState(const uint8_t& passIndex)
	{
		if (m_PassCache.size() == 0)
		{
			ApplyProperties();
		}

		GfxRenderState* passState = nullptr;
		if (passIndex < m_PassCache.size())
		{
			passState = &m_PassCache[passIndex];
			if (passState->isValid)
			{
				return passState;
			}
		}

		if (!m_Shader.IsValid() || m_Shader->GetState() != ObjectState::Default)
		{
			return nullptr;
		}

		GfxRenderState newState = {};
		auto shaderPass = m_Shader.Get()->GetData()->GetPass(passIndex);
		if (shaderPass == nullptr)
		{
			return nullptr;
		}

		uint32_t vertexFlags = 0;
		uint32_t fragmentFlags = 0;
		if (m_ActiveKeywords.size() > 0)
		{
			auto vertexKeywords = shaderPass->GetVertexKeywords();
			auto fragmentKeywords = shaderPass->GetFragmentKeywords();
			for (auto& keyword : m_ActiveKeywords)
			{
				for (int i = 0; i < vertexKeywords.size(); ++i)
				{
					if (vertexKeywords[i] == keyword)
					{
						vertexFlags |= 1 << i;
					}
				}
				for (int i = 0; i < fragmentKeywords.size(); ++i)
				{
					if (fragmentKeywords[i] == keyword)
					{
						fragmentFlags |= 1 << i;
						break;
					}
				}
			}
		}
		const ShaderVariant variant = m_Shader->GetVariant(vertexFlags, fragmentFlags, passIndex);
		newState.vertexShader = variant.vertexShader;
		newState.geometryShader = variant.geometryShader;
		newState.fragmentShader = variant.fragmentShader;

		auto textureSlots = variant.fragmentShader->m_TextureSlots;
		for (auto& slot : textureSlots)
		{
			auto it = m_TextureMap.find(slot.first);
			if (it != m_TextureMap.end())
			{
				GfxRenderState::TextureInfo info = {};
				info.textureId = &it->second;
				info.textureSlot = slot.second.first;
				info.samplerSlot = slot.second.second;
				newState.fragmentTextures[newState.fragmentTextureCount] = info;
				++newState.fragmentTextureCount;
			}
		}

		newState.cullMode = shaderPass->GetCullMode();
		newState.blendSrcColor = shaderPass->GetBlendSrcColor();
		newState.blendSrcAlpha = shaderPass->GetBlendSrcAlpha();
		newState.blendDstColor = shaderPass->GetBlendDstColor();
		newState.blendDstAlpha = shaderPass->GetBlendDstAlpha();
		newState.zTest = shaderPass->GetZTest();
		newState.zWrite = shaderPass->GetZWrite();

		newState.isValid = true;

		m_PassCache[passIndex] = newState;
		return passState;
	}

	const uint32_t& Material::GetCRC()
	{
		if (m_Crc == -1)
		{
			for (auto& pair : m_TextureMap)
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
				if (m_TextureMap.count(key) == 0)
				{
					m_TextureMap.insert({key, ((Texture*)DefaultTextures::GetTexture(data->GetDefaultTextureName()))->GetObjectId() });
				}
			}
		}
		for (auto const& texture : m_Textures)
		{
			TextureData* textureData = texture.Get();
			if (textureData->m_Texture.IsValid())
			{
				m_TextureMap.insert_or_assign(TO_HASH(textureData->m_Name), textureData->m_Texture->GetObjectId());
			}
		}
	}
}