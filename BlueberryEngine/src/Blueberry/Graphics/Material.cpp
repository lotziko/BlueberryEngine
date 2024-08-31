#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\DefaultTextures.h"
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
		material->OnCreate();
		return material;
	}

	void Material::SetTexture(std::size_t id, Texture* texture)
	{
		m_TextureMap.insert_or_assign(id, texture);
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
		GfxRenderState* passState = &m_PassCache[passIndex];
		if (passState->isValid)
		{
			return passState;
		}

		if (!m_Shader.IsValid() || m_Shader->GetState() != ObjectState::Default)
		{
			return nullptr;
		}

		GfxRenderState newState = {};
		auto shaderPass = m_Shader.Get()->GetData()->GetPass(passIndex);

		UINT vertexFlags = 0;
		UINT fragmentFlags = 0;
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
		const ShaderVariant variant = m_Shader->GetVariant(vertexFlags, fragmentFlags);
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
				info.texture = it->second.Get()->m_Texture;
				info.textureSlot = slot.second.first;
				info.samplerSlot = slot.second.second;
				newState.fragmentTextures[newState.fragmentTextureCount] = info;
				++newState.fragmentTextureCount;
			}
		}

		newState.cullMode = shaderPass->GetCullMode();
		newState.blendSrc = shaderPass->GetBlendSrc();
		newState.blendDst = shaderPass->GetBlendDst();
		newState.zWrite = shaderPass->GetZWrite();

		newState.isValid = true;

		m_PassCache[passIndex] = newState;
		return passState;
	}

	//const std::pair<UINT, UINT>& Material::GetKeywordFlags()
	//{
	//	// TODO move into pipeline state
	//	if (m_ActiveKeywords.size() == 0 || !m_Shader.IsValid())
	//	{
	//		return std::make_pair(0, 0);
	//	}
	//	auto pass = m_Shader.Get()->GetData()->GetPass(0);
	//	auto vertexKeywords = pass->GetVertexKeywords();
	//	auto fragmentKeywords = pass->GetFragmentKeywords();
	//	UINT vertexFlags = 0;
	//	UINT fragmentFlags = 0;
	//	for (auto& keyword : m_ActiveKeywords)
	//	{
	//		for (int i = 0; i < vertexKeywords.size(); ++i)
	//		{
	//			if (vertexKeywords[i] == keyword)
	//			{
	//				vertexFlags |= 1 << i;
	//			}
	//		}
	//		for (int i = 0; i < fragmentKeywords.size(); ++i)
	//		{
	//			if (fragmentKeywords[i] == keyword)
	//			{
	//				fragmentFlags |= 1 << i;
	//				break;
	//			}
	//		}
	//	}
	//	return std::make_pair(vertexFlags, fragmentFlags);
	//}

	void Material::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Material)
		BIND_FIELD(FieldInfo(TO_STRING(m_Shader), &Material::m_Shader, BindingType::ObjectPtr).SetObjectType(Shader::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Textures), &Material::m_Textures, BindingType::DataArray).SetObjectType(TextureData::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_ActiveKeywords), &Material::m_ActiveKeywords, BindingType::StringArray))
		END_OBJECT_BINDING()
	}

	void Material::OnCreate()
	{
		ApplyProperties();
	}

	void Material::FillTextureMap()
	{
		m_TextureMap.clear();
		if (m_Shader.IsValid() && m_Shader->GetState() == ObjectState::Default)
		{
			const ShaderData* shaderData = m_Shader->GetData();
			for (auto& parameter : shaderData->GetTextureParameters())
			{
				TextureParameterData* data = parameter.Get();
				m_TextureMap.insert_or_assign(TO_HASH(data->GetName()), (Texture*)DefaultTextures::GetTexture(data->GetDefaultTextureName()));
			}
		}
		for (auto const& texture : m_Textures)
		{
			TextureData* textureData = texture.Get();
			if (textureData->m_Texture.IsValid())
			{
				m_TextureMap.insert_or_assign(TO_HASH(textureData->m_Name), textureData->m_Texture);
			}
		}
	}
}