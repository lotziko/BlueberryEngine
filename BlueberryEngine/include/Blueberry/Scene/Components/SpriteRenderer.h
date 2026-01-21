#pragma once

#include "Renderer.h"

namespace Blueberry
{
	class Texture2D;
	class Material;

	class BB_API SpriteRenderer : public Renderer
	{
		OBJECT_DECLARATION(SpriteRenderer)

	public:
		SpriteRenderer() = default;
		virtual ~SpriteRenderer() = default;

		virtual const AABB& GetBounds() final;
		virtual const Matrix& GetLocalToWorldMatrix() final;

		const Color& GetColor();
		void SetColor(const Color& color);

		Texture2D* GetTexture();
		void SetTexture(Texture2D* texture);

		Material* GetMaterial();
		void SetMaterial(Material* material);

	private:
		AABB m_Bounds = AABB(Vector3::Zero, Vector3::Zero);
		Color m_Color = Color(1, 1, 1, 1);
		ObjectPtr<Texture2D> m_Texture;
		ObjectPtr<Material> m_Material;
	};
}