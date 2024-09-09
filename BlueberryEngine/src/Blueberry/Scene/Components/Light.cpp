#include "bbpch.h"
#include "Light.h"

#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Light)

	const LightType& Light::GetType()
	{
		return m_Type;
	}

	const Color& Light::GetColor()
	{
		return m_Color;
	}

	const float& Light::GetIntensity()
	{
		return m_Intensity;
	}

	const float& Light::GetRange()
	{
		return m_Range;
	}

	const float& Light::GetOuterSpotAngle()
	{
		return m_OuterSpotAngle;
	}

	const float& Light::GetInnerSpotAngle()
	{
		return m_InnerSpotAngle;
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
		END_OBJECT_BINDING()
	}
}