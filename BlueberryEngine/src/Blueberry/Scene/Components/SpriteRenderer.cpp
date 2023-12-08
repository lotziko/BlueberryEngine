#include "bbpch.h"
#include "SpriteRenderer.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, SpriteRenderer)

	SpriteRenderer::SpriteRenderer()
	{
		m_Color = Color(1, 1, 1, 1);
		auto type = Texture2D::Type;
		m_Texture = nullptr;
		m_Material = nullptr;
	}

	const Color& SpriteRenderer::GetColor()
	{
		return m_Color;
	}

	void SpriteRenderer::SetColor(const Color& color)
	{
		m_Color = color;
	}

	const Ref<Texture2D>& SpriteRenderer::GetTexture()
	{
		return m_Texture;
	}

	void SpriteRenderer::SetTexture(const Ref<Texture2D>& texture)
	{
		m_Texture = texture;
	}

	const Ref<Material>& SpriteRenderer::GetMaterial()
	{
		return m_Material;
	}

	void SpriteRenderer::SetMaterial(const Ref<Material>& material)
	{
		m_Material = material;
	}

	void SpriteRenderer::BindProperties()
	{
		BEGIN_OBJECT_BINDING(SpriteRenderer)
		BIND_FIELD("m_Color", &SpriteRenderer::m_Color, BindingType::Color)
		END_OBJECT_BINDING()
	}
}