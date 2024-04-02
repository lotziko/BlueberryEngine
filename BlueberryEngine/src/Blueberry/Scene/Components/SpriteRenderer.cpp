#include "bbpch.h"
#include "SpriteRenderer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, SpriteRenderer)

	const Color& SpriteRenderer::GetColor()
	{
		return m_Color;
	}

	void SpriteRenderer::SetColor(const Color& color)
	{
		m_Color = color;
	}

	Texture2D* SpriteRenderer::GetTexture()
	{
		return m_Texture.Get();
	}

	void SpriteRenderer::SetTexture(Texture2D* texture)
	{
		m_Texture = texture;
	}

	Material* SpriteRenderer::GetMaterial()
	{
		return m_Material.Get();
	}

	void SpriteRenderer::SetMaterial(Material* material)
	{
		m_Material = material;
	}

	void SpriteRenderer::BindProperties()
	{
		BEGIN_OBJECT_BINDING(SpriteRenderer)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &SpriteRenderer::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Color), &SpriteRenderer::m_Color, BindingType::Color))
		BIND_FIELD(FieldInfo(TO_STRING(m_Texture), &SpriteRenderer::m_Texture, BindingType::ObjectPtr).SetObjectType(Texture::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Material), &SpriteRenderer::m_Material, BindingType::ObjectPtr).SetObjectType(Material::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_SortingOrder), &SpriteRenderer::m_SortingOrder, BindingType::Int))
		END_OBJECT_BINDING()
	}
}