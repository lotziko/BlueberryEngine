#include "bbpch.h"
#include "Material.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Object, Material)

	Material* Material::Create(Shader* shader)
	{
		Material* material = Object::Create<Material>();
		material->SetShader(shader);
		material->OnCreate();
		return material;
	}

	void Material::SetTexture(std::size_t id, Texture* texture)
	{
		if (m_Textures.count(id) == 0)
		{
			m_Textures.insert({ id, ObjectPtr<Texture>(texture) });
			FillGfxTextures();
		}
	}

	void Material::SetTexture(std::string name, Texture* texture)
	{
		SetTexture(std::hash<std::string>()(name), texture);
	}

	void Material::SetShader(Shader* shader)
	{
		m_Shader = shader;
	}

	const ShaderOptions& Material::GetShaderOptions()
	{
		if (m_Shader.IsValid())
		{
			return m_Shader.Get()->GetOptions();
		}
		return ShaderOptions();
	}

	void Material::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Material)
		BIND_FIELD(FieldInfo(TO_STRING(m_Shader), &Material::m_Shader, BindingType::ObjectPtr).SetObjectType(Shader::Type))
		//BIND_FIELD(FieldInfo(TO_STRING(m_CullMode), &Material::m_CullMode, BindingType::Enum).SetHintData("None,Front,Back"))
		//BIND_FIELD(FieldInfo(TO_STRING(m_SurfaceType), &Material::m_SurfaceType, BindingType::Enum).SetHintData("Opaque,Transparent,DepthTransparent"))
		END_OBJECT_BINDING()
	}

	void Material::FillGfxTextures()
	{
		m_GfxTextures.clear();
		std::map<std::size_t, ObjectPtr<Texture>>::iterator it;
		for (it = m_Textures.begin(); it != m_Textures.end(); it++)
		{
			m_GfxTextures.emplace_back(std::make_pair(it->first, it->second.Get()->m_Texture));
		}
	}
}