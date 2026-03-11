#include "Blueberry\Scene\Components\Light.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "..\..\Graphics\LightHelper.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Light, Component)
	{
		DEFINE_BASE_FIELDS(Light, Component)
		DEFINE_FIELD(Light, m_Type, BindingType::Enum, FieldOptions().SetEnumHint("Spot,Directional,Point"))
		DEFINE_FIELD(Light, m_Color, BindingType::Color, {})
		DEFINE_FIELD(Light, m_Intensity, BindingType::Float, {})
		DEFINE_FIELD(Light, m_Range, BindingType::Float, {})
		DEFINE_FIELD(Light, m_OuterSpotAngle, BindingType::Float, {})
		DEFINE_FIELD(Light, m_InnerSpotAngle, BindingType::Float, {})
		DEFINE_FIELD(Light, m_IsCastingShadows, BindingType::Bool, {})
		DEFINE_FIELD(Light, m_IsCastingFog, BindingType::Bool, {})
		DEFINE_FIELD(Light, m_IsCached, BindingType::Bool, {})
		DEFINE_FIELD(Light, m_Cookie, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture::Type))
		DEFINE_ITERATOR(Light)
		DEFINE_EXECUTE_ALWAYS()
	}

	void Light::OnDisable()
	{
		ReleaseCachedShadow();
	}

	void Light::OnPreCull()
	{
		UpdateBounds();
	}

	const LightType& Light::GetType()
	{
		return m_Type;
	}

	void Light::SetType(const LightType& type)
	{
		m_Type = type;
	}

	const Color& Light::GetColor()
	{
		return m_Color;
	}

	void Light::SetColor(const Color& color)
	{
		m_Color = color;
	}

	const float& Light::GetIntensity()
	{
		return m_Intensity;
	}

	void Light::SetIntensity(const float& intensity)
	{
		m_Intensity = intensity;
	}

	const float& Light::GetRange()
	{
		return m_Range;
	}

	void Light::SetRange(const float& range)
	{
		m_Range = range;
	}

	const float& Light::GetOuterSpotAngle()
	{
		return m_OuterSpotAngle;
	}

	const float& Light::GetInnerSpotAngle()
	{
		return m_InnerSpotAngle;
	}

	const bool& Light::IsCastingShadows()
	{
		return m_IsCastingShadows;
	}

	void Light::SetCastingShadows(const bool& castingShadows)
	{
		m_IsCastingShadows = castingShadows;
	}

	const bool& Light::IsCastingFog()
	{
		return m_IsCastingFog;
	}

	void Light::SetCastingFog(const bool& castingFog)
	{
		m_IsCastingFog = castingFog;
	}

	const bool& Light::IsCached()
	{
		return m_IsCached;
	}

	void Light::SetCached(const bool& cached)
	{
		m_IsCached = cached;
	}

	Texture* Light::GetCookie()
	{
		return m_Cookie.Get();
	}

	void Light::SetCookie(Texture* cookie)
	{
		m_Cookie = cookie;
	}

	GfxTexture* Light::GetCachedShadow()
	{
		if (m_CachedShadow == nullptr)
		{
			uint32_t size = LightHelper::GetShadowSize(m_Type);
			uint32_t sliceCount = LightHelper::GetSliceCount(m_Type);
			m_CachedShadow = GfxTexturePool::Get(size * sliceCount, size, 1, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::D32_Float, TextureDimension::Texture2D, WrapMode::Clamp, FilterMode::Point);
		}
		return m_CachedShadow;
	}

	void Light::ReleaseCachedShadow()
	{
		if (m_CachedShadow != nullptr)
		{
			GfxTexturePool::Release(m_CachedShadow);
			m_CachedShadow = nullptr;
			memset(m_IsDirty, true, sizeof(bool) * 6);
		}
	}

	void Light::UpdateBounds()
	{
		Transform* transform = GetTransform();
		size_t transformUpdateCount = transform->GetUpdateCount();
		if (m_UpdateCount < transformUpdateCount)
		{
			memset(m_IsDirty, true, sizeof(bool) * 6);
			m_UpdateCount = transformUpdateCount;
		}
	}
}