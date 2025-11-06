#include "SceneSettings.h"

#include "Blueberry\Core\ClassDB.h"
#include "LightingData.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SceneSettings, Object)
	{
		DEFINE_FIELD(SceneSettings, m_LightingData, BindingType::ObjectPtr, FieldOptions().SetObjectType(LightingData::Type))
	}

	LightingData* SceneSettings::GetLightingData()
	{
		return m_LightingData.Get();
	}

	void SceneSettings::SetLightingData(LightingData* data)
	{
		m_LightingData = data;
	}
}