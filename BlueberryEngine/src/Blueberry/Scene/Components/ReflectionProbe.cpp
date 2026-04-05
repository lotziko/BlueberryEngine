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
		DEFINE_FIELD(ReflectionProbe, m_AtlasIndex, BindingType::Uint, FieldOptions().SetSerializationFlags(SerializationFlags::RuntimeOnly))
		DEFINE_ITERATOR(ReflectionProbe)
	}

	ReflectionProbeType ReflectionProbe::GetType()
	{
		return m_Type;
	}

	void ReflectionProbe::SetType(ReflectionProbeType type)
	{
		m_Type = type;
	}

	uint32_t ReflectionProbe::GetAtlasIndex() const
	{
		return m_AtlasIndex;
	}

	void ReflectionProbe::SetAtlasIndex(uint32_t atlasIndex)
	{
		m_AtlasIndex = atlasIndex;
	}

	float ReflectionProbe::GetRadius() const
	{
		return m_Radius;
	}

	const Vector3& ReflectionProbe::GetSize() const
	{
		return m_Size;
	}
}