#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Material)

	Material* Material::Create(Shader* shader)
	{
		Material* material = Object::Create<Material>();
		material->m_Shader = shader;
		return material;
	}

	void Material::SetTexture(std::size_t id, Texture* texture)
	{
		if (m_Textures.count(id) == 0)
		{
			m_Textures.insert({ id, WeakObjectPtr<Texture>(texture) });
			FillGfxTextures();
		}
	}

	void Material::SetTexture(std::string name, Texture* texture)
	{
		SetTexture(std::hash<std::string>()(name), texture);
	}

	void Material::BindProperties()
	{
	}

	void Material::FillGfxTextures()
	{
		m_GfxTextures.clear();
		std::map<std::size_t, WeakObjectPtr<Texture>>::iterator it;
		for (it = m_Textures.begin(); it != m_Textures.end(); it++)
		{
			m_GfxTextures.emplace_back(std::make_pair(it->first, it->second.Get()->m_Texture));
		}
	}
}