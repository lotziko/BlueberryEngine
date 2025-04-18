#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Material;

	class SkyRenderer : public Component
	{
		OBJECT_DECLARATION(SkyRenderer)

	public:
		SkyRenderer() = default;
		virtual ~SkyRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		Material* GetMaterial();

	private:
		ObjectPtr<Material> m_Material;
	};
}