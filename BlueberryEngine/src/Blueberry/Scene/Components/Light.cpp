#include "bbpch.h"
#include "Light.h"

#include "Blueberry\Scene\Entity.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Light)

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

	void Light::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Light)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &Light::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Color), &Light::m_Color, BindingType::Color))
		BIND_FIELD(FieldInfo(TO_STRING(m_Intensity), &Light::m_Intensity, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_Range), &Light::m_Range, BindingType::Float))
		END_OBJECT_BINDING()
	}
}