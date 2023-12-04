#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Material)

	Material::Material(const Ref<Shader>& shader)
	{
		m_Shader = shader;
	}

	Ref<Material> Material::Create(const Ref<Shader>& shader)
	{
		return ObjectDB::CreateObject<Material>(shader);
	}

	void Material::SetTexture(std::size_t id, const Ref<Texture>& texture)
	{
		if (m_Textures.count(id) == 0)
		{
			m_Textures.insert({ id, texture });
			FillGfxTextures();
		}
	}

	void Material::SetTexture(std::string name, const Ref<Texture>& texture)
	{
		SetTexture(std::hash<std::string>()(name), texture);
	}

	void Material::BindProperties()
	{
	}

	void Material::FillGfxTextures()
	{
		m_GfxTextures.clear();
		std::map<std::size_t, Ref<Texture>>::iterator it;
		for (it = m_Textures.begin(); it != m_Textures.end(); it++)
		{
			m_GfxTextures.emplace_back(std::make_pair(it->first, it->second->m_Texture.get()));
		}
	}
}