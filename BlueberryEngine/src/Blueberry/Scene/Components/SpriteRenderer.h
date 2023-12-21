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

		Texture2D* GetTexture();
		void SetTexture(Texture2D* texture);

		Material* GetMaterial();
		void SetMaterial(Material* material);

		static void BindProperties();

	private:
		Color m_Color;
		WeakObjectPtr<Texture2D> m_Texture;
		WeakObjectPtr<Material> m_Material;
	};
}