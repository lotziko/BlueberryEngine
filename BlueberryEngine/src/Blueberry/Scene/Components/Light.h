#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	enum class LightType
	{
		Spot,
		Directional,
		Point
	};

	class Light : public Component
	{
		OBJECT_DECLARATION(Light)

	public:
		Light() = default;
		virtual ~Light() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		const LightType& GetType();
		void SetType(const LightType& type);

		const Color& GetColor();
		void SetColor(const Color& color);

		const float& GetIntensity();
		void SetIntensity(const float& intensity);

		const float& GetRange();
		void SetRange(const float& range);

		const float& GetOuterSpotAngle();
		const float& GetInnerSpotAngle();

		static void BindProperties();

	private:
		LightType m_Type = LightType::Point;
		Color m_Color = Color(1.0f, 1.0f, 1.0f, 1.0f);
		float m_Intensity = 1.0f;
		float m_Range = 5.0f;
		float m_OuterSpotAngle = 30.0f;
		float m_InnerSpotAngle = 15.0f;
	};
}