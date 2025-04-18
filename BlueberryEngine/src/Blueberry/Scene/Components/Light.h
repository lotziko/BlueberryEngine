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

		const bool& IsCastingShadows();
		void SetCastingShadows(const bool& castingShadows);

	private:
		LightType m_Type = LightType::Point;
		Color m_Color = Color(1.0f, 1.0f, 1.0f, 1.0f);
		float m_Intensity = 1.0f;
		float m_Range = 5.0f;
		float m_OuterSpotAngle = 30.0f;
		float m_InnerSpotAngle = 15.0f;
		bool m_IsCastingShadows = true;

	private:
		uint8_t m_SliceCount = 1;
		Matrix m_WorldToShadow[6];
		Matrix m_AtlasWorldToShadow[6];
		Vector4 m_ShadowBounds[6];
		Vector4 m_ShadowCascades[6];

		friend class RenderContext;
		friend class ShadowAtlas;
		friend class PerCameraLightDataConstantBuffer;
	};
}