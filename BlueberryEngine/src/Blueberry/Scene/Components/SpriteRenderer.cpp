#include "bbpch.h"
#include "SpriteRenderer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SpriteRenderer, Renderer)
	{
		DEFINE_BASE_FIELDS(SpriteRenderer, Renderer)
		DEFINE_FIELD(SpriteRenderer, m_Color, BindingType::Color, {})
		DEFINE_FIELD(SpriteRenderer, m_Texture, BindingType::ObjectPtr, FieldOptions().SetObjectType(Texture::Type))
		DEFINE_FIELD(SpriteRenderer, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(Material::Type))
		DEFINE_FIELD(SpriteRenderer, m_SortingOrder, BindingType::Int, {})
	}

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
}