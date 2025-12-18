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

		Material* GetMaterial();

		const Color& GetAmbientColor();
		void SetAmbientColor(const Color& color);

	private:
		ObjectPtr<Material> m_Material;
		Color m_AmbientColor;
	};
}