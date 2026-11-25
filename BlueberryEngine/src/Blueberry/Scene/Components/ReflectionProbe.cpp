#include "Blueberry\Scene\Components\ReflectionProbe.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ReflectionProbe, Component)
	{
		DEFINE_BASE_FIELDS(ReflectionProbe, Component)
		DEFINE_FIELD(ReflectionProbe, m_Type, BindingType::Enum, FieldOptions().SetEnumHint("Sphere,Box"))
		DEFINE_FIELD(ReflectionProbe, m_Radius, BindingType::Float, FieldOptions())
		DEFINE_FIELD(ReflectionProbe, m_Size, BindingType::Vector3, FieldOptions())
		DEFINE_FIELD(ReflectionProbe, m_Fade, BindingType::Float, FieldOptions())
		DEFINE_FIELD(ReflectionProbe, m_Index, BindingType::Uint, FieldOptions().SetSerializationFlags(SerializationFlags::RuntimeOnly))
		DEFINE_ITERATOR(ReflectionProbe)
		DEFINE_EXECUTE_ALWAYS()
		DEFINE_ADD_COMPONENT_MENU("Rendering/Reflection Probe")
	}

	ReflectionProbeType ReflectionProbe::GetType()
	{
		return m_Type;
	}

	void ReflectionProbe::SetType(ReflectionProbeType type)
	{
		m_Type = type;
	}

	uint32_t ReflectionProbe::GetIndex() const
	{
		return m_Index;
	}

	void ReflectionProbe::SetIndex(uint32_t index)
	{
		m_Index = index;
	}

	float ReflectionProbe::GetRadius() const
	{
		return m_Radius;
	}

	void ReflectionProbe::SetRadius(float radius)
	{
		m_Radius = radius;
	}

	const Vector3& ReflectionProbe::GetSize() const
	{
		return m_Size;
	}

	void ReflectionProbe::SetSize(const Vector3& size)
	{
		m_Size = size;
	}

	float ReflectionProbe::GetFade() const
	{
		return m_Fade;
	}
}