#include "bbpch.h"
#include "SpriteRenderer.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, SpriteRenderer)

	SpriteRenderer::SpriteRenderer()
	{
		m_Color = Color(1, 1, 1, 1);
		g_AssetManager->Load<Texture>("assets/TestImage", m_Texture);
	}

	const Color& SpriteRenderer::GetColor()
	{
		return m_Color;
	}

	void SpriteRenderer::SetColor(const Color& color)
	{
		m_Color = color;
	}

	const Ref<Texture>& SpriteRenderer::GetTexture()
	{
		return m_Texture;
	}

	void SpriteRenderer::SetTexture(const Ref<Texture>& texture)
	{
		m_Texture = texture;
	}

	std::string SpriteRenderer::ToString() const
	{
		return "SpriteRenderer";
	}
}