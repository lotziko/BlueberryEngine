#include "bbpch.h"
#include "Light.h"

#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Light)

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

	void Light::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Light)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &Light::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Type), &Light::m_Type, BindingType::Enum).SetHintData("Spot,Directional,Point"))
		BIND_FIELD(FieldInfo(TO_STRING(m_Color), &Light::m_Color, BindingType::Color))
		BIND_FIELD(FieldInfo(TO_STRING(m_Intensity), &Light::m_Intensity, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_Range), &Light::m_Range, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_OuterSpotAngle), &Light::m_OuterSpotAngle, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_InnerSpotAngle), &Light::m_InnerSpotAngle, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_IsCastingShadows), &Light::m_IsCastingShadows, BindingType::Bool))
		END_OBJECT_BINDING()
	}
}