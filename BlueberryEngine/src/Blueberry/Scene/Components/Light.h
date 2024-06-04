#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Light : public Component
	{
		OBJECT_DECLARATION(Light)

	public:
		Light() = default;
		virtual ~Light() = default;

		const Color& GetColor();
		const float& GetIntensity();
		const float& GetRange();

		static void BindProperties();

	private:
		Color m_Color = Color(1.0f, 1.0f, 1.0f, 1.0f);
		float m_Intensity = 1.0f;
		float m_Range = 5.0f;
	};
}