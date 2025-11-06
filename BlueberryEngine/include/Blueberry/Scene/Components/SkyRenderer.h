#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Material;
	class TextureCube;

	class BB_API SkyRenderer : public Component
	{
		OBJECT_DECLARATION(SkyRenderer)

	public:
		SkyRenderer() = default;
		virtual ~SkyRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		Material* GetMaterial();

		const Color& GetAmbientColor();
		void SetAmbientColor(const Color& color);

		TextureCube* GetReflectionTexture();
		void SetReflectionTexture(TextureCube* texture);

	private:
		ObjectPtr<Material> m_Material;
		Color m_AmbientColor;
		ObjectPtr<TextureCube> m_ReflectionTexture;
	};
}