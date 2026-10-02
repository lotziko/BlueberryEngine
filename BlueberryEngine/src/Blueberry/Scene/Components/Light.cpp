#include "Blueberry\Scene\Components\Light.h"

#include "Blueberry\Scene\Scene.h"
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
		DEFINE_FIELD(Light, m_Type, BindingType::Enum, FieldOptions().SetEnumHint("Spot,Directional,Point").SetUpdateCallback(MethodBind::Create(&Light::ReleaseCachedShadow)))
		DEFINE_FIELD(Light, m_Color, BindingType::Color, FieldOptions())
		DEFINE_FIELD(Light, m_Intensity, BindingType::Float, FieldOptions())
		DEFINE_FIELD(Light, m_Range, BindingType::Float, FieldOptions().SetUpdateCallback(MethodBind::Create(&Light::InvalidateCache)))
		DEFINE_FIELD(Light, m_OuterSpotAngle, BindingType::Float, FieldOptions().SetUpdateCallback(MethodBind::Create(&Light::InvalidateCache)))
		DEFINE_FIELD(Light, m_InnerSpotAngle, BindingType::Float, FieldOptions().SetUpdateCallback(MethodBind::Create(&Light::InvalidateCache)))
		DEFINE_FIELD(Light, m_IsCastingShadows, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(Light, m_IsCastingFog, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(Light, m_IsCached, BindingType::Bool, FieldOptions().SetUpdateCallback(MethodBind::Create(&Light::InvalidateCache)))
		DEFINE_FIELD(Light, m_Cookie, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture::Type))
		DEFINE_ITERATOR(Light)
		DEFINE_EXECUTE_ALWAYS()
	}

	void Light::OnEnable()
	{
		GetTransform()->AddDependency(this);
	}

	void Light::OnDisable()
	{
		GetTransform()->RemoveDependency(this);
		ReleaseCachedShadow();
	}

	void Light::OnTransformInvalidate()
	{
		InvalidateCache();
	}

	LightType Light::GetType()
	{
		return m_Type;
	}

	void Light::SetType(LightType type)
	{
		m_Type = type;
		ReleaseCachedShadow();
	}

	const Color& Light::GetColor() const
	{
		return m_Color;
	}

	void Light::SetColor(const Color& color)
	{
		m_Color = color;
	}

	float Light::GetIntensity() const
	{
		return m_Intensity;
	}

	void Light::SetIntensity(float intensity)
	{
		m_Intensity = intensity;
	}

	float Light::GetRange() const
	{
		return m_Range;
	}

	void Light::SetRange(float range)
	{
		m_Range = range;
		InvalidateCache();
	}

	float Light::GetOuterSpotAngle() const
	{
		return m_OuterSpotAngle;
	}

	void Light::SetOuterSpotAngle(float outerSpotAngle)
	{
		m_OuterSpotAngle = outerSpotAngle;
		InvalidateCache();
	}

	float Light::GetInnerSpotAngle() const
	{
		return m_InnerSpotAngle;
	}

	void Light::SetInnerSpotAngle(float innerSpotAngle)
	{
		m_InnerSpotAngle = innerSpotAngle;
		InvalidateCache();
	}

	bool Light::IsCastingShadows() const
	{
		return m_IsCastingShadows;
	}

	void Light::SetCastingShadows(bool castingShadows)
	{
		m_IsCastingShadows = castingShadows;
	}

	bool Light::IsCastingFog() const
	{
		return m_IsCastingFog;
	}

	void Light::SetCastingFog(bool castingFog)
	{
		m_IsCastingFog = castingFog;
	}

	bool Light::IsCached() const
	{
		return m_IsCached;
	}

	void Light::SetCached(bool cached)
	{
		m_IsCached = cached;
		InvalidateCache();
	}

	Texture* Light::GetCookie() const
	{
		return m_Cookie.Get();
	}

	void Light::SetCookie(Texture* cookie)
	{
		m_Cookie = cookie;
	}

	void Light::InvalidateCache()
	{
		if (m_IsCached)
		{
			memset(m_IsDirty, true, sizeof(bool) * 6);
		}
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
}