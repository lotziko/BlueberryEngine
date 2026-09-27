#pragma once

#include "Blueberry\Core\Base.h"
#include "AutoExposure.h"
#include "VolumetricFog.h"
#include "Reflections.h"

namespace Blueberry
{
	class PerCameraData
	{
	private:
		PerCameraExposureData m_ExposureData;
		PerCameraVolumetricFogData m_VolumetricFogData;
		PerCameraReflectionsData m_ReflectionsData;

		friend class AutoExposure;
		friend class VolumetricFog;
		friend class Reflections;
	};
}