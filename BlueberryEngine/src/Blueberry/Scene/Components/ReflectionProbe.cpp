#include "Blueberry\Scene\Components\ReflectionProbe.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ReflectionProbe, Component)
	{
		DEFINE_BASE_FIELDS(ReflectionProbe, Component)
		DEFINE_FIELD(ReflectionProbe, m_Type, BindingType::Enum, FieldOptions().SetEnumHint("Sphere,Box"))
		DEFINE_FIELD(ReflectionProbe, m_Radius, BindingType::Float, {})
		DEFINE_FIELD(ReflectionProbe, m_Size, BindingType::Vector3, {})
		DEFINE_FIELD(ReflectionProbe, m_Fade, BindingType::Float, {})
	}

	void ReflectionProbe::OnEnable()
	{
		AddToSceneComponents(ReflectionProbe::Type);
	}

	void ReflectionProbe::OnDisable()
	{
		RemoveFromSceneComponents(ReflectionProbe::Type);
	}

	const ReflectionProbeType& ReflectionProbe::GetType()
	{
		return m_Type;
	}

	void ReflectionProbe::SetType(const ReflectionProbeType& type)
	{
		m_Type = type;
	}

	const uint32_t& ReflectionProbe::GetAtlasIndex()
	{
		return m_AtlasIndex;
	}

	void ReflectionProbe::SetAtlasIndex(const uint32_t& atlasIndex)
	{
		m_AtlasIndex = atlasIndex;
	}

	const float& ReflectionProbe::GetRadius()
	{
		return m_Radius;
	}

	const Vector3& ReflectionProbe::GetSize()
	{
		return m_Size;
	}
}