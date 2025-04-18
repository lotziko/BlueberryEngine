#include "bbpch.h"
#include "Light.h"

#include "Blueberry\Scene\Entity.h"

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
	}

	void Light::OnEnable()
	{
		AddToSceneComponents(Light::Type);
	}

	void Light::OnDisable()
	{
		RemoveFromSceneComponents(Light::Type);
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
}