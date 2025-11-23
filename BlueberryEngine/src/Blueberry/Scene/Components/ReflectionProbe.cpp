#include "Blueberry\Scene\Components\ReflectionProbe.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(ReflectionProbe, Component)
	{
		DEFINE_BASE_FIELDS(ReflectionProbe, Component)
		DEFINE_FIELD(ReflectionProbe, m_ReflectionTexture, BindingType::ObjectPtr, FieldOptions().SetObjectType(TextureCube::Type))
		DEFINE_FIELD(ReflectionProbe, m_Size, BindingType::Vector3, {})
	}

	void ReflectionProbe::OnEnable()
	{
		AddToSceneComponents(ReflectionProbe::Type);
	}

	void ReflectionProbe::OnDisable()
	{
		RemoveFromSceneComponents(ReflectionProbe::Type);
	}

	TextureCube* ReflectionProbe::GetReflectionTexture()
	{
		return m_ReflectionTexture.Get();
	}

	void ReflectionProbe::SetReflectionTexture(TextureCube* texture)
	{
		m_ReflectionTexture = texture;
	}

	void ReflectionProbe::SetAtlasIndex(const uint32_t& atlasIndex)
	{
		m_AtlasIndex = atlasIndex;
	}

	const Vector3& ReflectionProbe::GetSize()
	{
		return m_Size;
	}
}