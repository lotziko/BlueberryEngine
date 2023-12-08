#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Renderer.h"

namespace Blueberry
{
	class Texture2D;
	class Material;

	class SpriteRenderer : public Renderer
	{
		OBJECT_DECLARATION(SpriteRenderer)

	public:
		SpriteRenderer();
		~SpriteRenderer() = default;

		const Color& GetColor();
		void SetColor(const Color& color);

		const Ref<Texture2D>& GetTexture();
		void SetTexture(const Ref<Texture2D>& texture);

		const Ref<Material>& GetMaterial();
		void SetMaterial(const Ref<Material>& material);

		static void BindProperties();

	private:
		Color m_Color;
		Ref<Texture2D> m_Texture;
		Ref<Material> m_Material;
	};
}