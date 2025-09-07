#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class LightingData;

	class SceneSettings : public Object
	{
		OBJECT_DECLARATION(SceneSettings)

	public:
		SceneSettings() = default;
		virtual ~SceneSettings() = default;

		LightingData* GetLightingData();
		void SetLightingData(LightingData* data);

	private:
		ObjectPtr<LightingData> m_LightingData;
	};
}