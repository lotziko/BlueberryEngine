#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Renderer.h"

namespace Blueberry
{
	class Texture;

	class SpriteRenderer : public Renderer
	{
		OBJECT_DECLARATION(SpriteRenderer)
		COMPONENT_DECLARATION(SpriteRenderer)

	public:
		SpriteRenderer();
		~SpriteRenderer() = default;

		const Color& GetColor();
		void SetColor(const Color& color);

		const Ref<Texture>& GetTexture();
		void SetTexture(const Ref<Texture>& texture);

		virtual std::string ToString() const final;

	private:
		Color m_Color;
		Ref<Texture> m_Texture;
	};
}