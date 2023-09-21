#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"

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

	void Material::SetTexture(const Ref<Texture>& texture)
	{
		m_Texture = texture;
	}
}