#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Renderer.h"

namespace Blueberry
{
	class Texture;

	class SpriteRenderer : public Renderer
	{
		OBJECT_DECLARATION(SpriteRenderer)

	public:
		SpriteRenderer()
		{
			m_Color = Color(1, 1, 1, 1);
		}
		~SpriteRenderer() = default;

		const Color& GetColor() { return m_Color; }
		void SetColor(const Color& color) { m_Color = color; }

		const Ref<Texture>& GetTexture() { return m_Texture; }
		void SetTexture(const Ref<Texture>& texture) { m_Texture = texture; }

	private:
		Color m_Color;
		Ref<Texture> m_Texture;
	};
}