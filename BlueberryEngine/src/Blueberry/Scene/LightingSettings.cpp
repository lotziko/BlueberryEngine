#include "Blueberry\Scene\LightingSettings.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\TextureCubeArray.h"

namespace Blueberry
{
	OBJECT_DEFINITION(LightingSettings, Object)
	{
		DEFINE_FIELD(LightingSettings, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture2D::Type))
		DEFINE_FIELD(LightingSettings, m_ProbeVolume, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture3D::Type))
		DEFINE_FIELD(LightingSettings, m_ReflectionProbes, BindingType::ObjectPtr, FieldOptions().SetObjectType(&TextureCubeArray::Type))
		DEFINE_FIELD(LightingSettings, m_ChartOffsetScale, BindingType::Vector4List, FieldOptions())
	}

	Texture2D* LightingSettings::GetLightmap()
	{
		return m_Lightmap.Get();
	}

	void LightingSettings::SetLightmap(Texture2D* lightmap)
	{
		m_Lightmap = lightmap;
	}

	Texture3D* LightingSettings::GetProbeVolume()
	{
		return m_ProbeVolume.Get();
	}

	void LightingSettings::SetProbeVolume(Texture3D* probeVolume)
	{
		m_ProbeVolume = probeVolume;
	}

	TextureCubeArray* LightingSettings::GetReflectionProbes()
	{
		return m_ReflectionProbes.Get();
	}

	void LightingSettings::SetReflectionProbes(TextureCubeArray* reflectionProbes)
	{
		m_ReflectionProbes = reflectionProbes;
	}

	List<Vector4>& LightingSettings::GetChartOffsetScale()
	{
		return m_ChartOffsetScale;
	}

	void LightingSettings::SetChartOffsetScale(const List<Vector4>& chartOffsetScale)
	{
		m_ChartOffsetScale = chartOffsetScale;
	}
}