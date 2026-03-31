#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	enum class BB_API LightType
	{
		Spot,
		Directional,
		Point
	};

	class Texture;
	class GfxTexture;

	class BB_API Light : public Component
	{
		OBJECT_DECLARATION(Light)

	public:
		Light() = default;
		virtual ~Light() = default;

		virtual void OnDisable() final;

		void OnPreCull();

		LightType GetType();
		void SetType(LightType type);

		const Color& GetColor() const;
		void SetColor(const Color& color);

		float GetIntensity() const;
		void SetIntensity(float intensity);

		float GetRange() const;
		void SetRange(float range);

		float GetOuterSpotAngle() const;
		float GetInnerSpotAngle() const;

		bool IsCastingShadows() const;
		void SetCastingShadows(bool castingShadows);

		bool IsCastingFog() const;
		void SetCastingFog(bool castingFog);

		bool IsCached() const;
		void SetCached(bool cached);

		Texture* GetCookie() const;
		void SetCookie(Texture* cookie);

	private:
		GfxTexture* GetCachedShadow();
		void ReleaseCachedShadow();

		void UpdateBounds();

	private:
		LightType m_Type = LightType::Point;
		Color m_Color = Color(1.0f, 1.0f, 1.0f, 1.0f);
		float m_Intensity = 1.0f;
		float m_Range = 5.0f;
		float m_OuterSpotAngle = 30.0f;
		float m_InnerSpotAngle = 15.0f;
		bool m_IsCastingShadows = true;
		bool m_IsCastingFog = true;
		bool m_IsCached = true;
		ObjectPtr<Texture> m_Cookie;

	private:
		GfxTexture* m_CachedShadow = nullptr;
		size_t m_UpdateCount = 0;
		bool m_IsDirty[6] = { true, true, true, true, true, true };
		Matrix m_WorldToShadow[6];
		Matrix m_AtlasWorldToShadow[6];
		Vector4 m_ShadowBounds[6];
		Vector4 m_ShadowCascades[6];
		Matrix m_WorldToCookie;

		friend class RenderContext;
		friend class ShadowAtlas;
		friend class CookieAtlas;
		friend class PerCameraLightDataConstantBuffer;
		friend class FogLightDataConstantBuffer;
	};
}