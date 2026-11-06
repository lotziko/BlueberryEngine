#include "Blueberry\Scene\LightingSettings.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"

namespace Blueberry
{
	OBJECT_DEFINITION(LightingSettings, Object)
	{
		DEFINE_FIELD(LightingSettings, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture2D::Type))
		DEFINE_FIELD(LightingSettings, m_ProbeVolumes, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&ProbeVolumeInfo::Type))
		DEFINE_FIELD(LightingSettings, m_ReflectionProbes, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&TextureCube::Type))
		DEFINE_FIELD(LightingSettings, m_ChartOffsetScale, BindingType::Vector4List, FieldOptions())
	}

	Texture2D* LightingSettings::GetLightmap() const
	{
		return m_Lightmap.Get();
	}

	void LightingSettings::SetLightmap(Texture2D* lightmap)
	{
		m_Lightmap = lightmap;
	}

	List<Vector4>& LightingSettings::GetChartOffsetScale()
	{
		return m_ChartOffsetScale;
	}

	void LightingSettings::SetChartOffsetScale(const List<Vector4>& chartOffsetScale)
	{
		m_ChartOffsetScale = chartOffsetScale;
	}

	const List<ObjectPtr<ProbeVolumeInfo>>& LightingSettings::GetProbeVolumes() const
	{
		return m_ProbeVolumes;
	}

	void LightingSettings::SetProbeVolumes(const List<ObjectPtr<ProbeVolumeInfo>>& probeVolumes)
	{
		m_ProbeVolumes = probeVolumes;
	}

	const List<ObjectPtr<TextureCube>>& LightingSettings::GetReflectionProbes() const
	{
		return m_ReflectionProbes;
	}

	void LightingSettings::SetReflectionProbes(const List<ObjectPtr<TextureCube>>& reflectionProbes)
	{
		m_ReflectionProbes = reflectionProbes;
	}

	uint32_t LightingSettings::GetLightmapUpdateCount() const
	{
		return m_LightmapUpdateCount;
	}

	void LightingSettings::IncrementLightmapUpdateCount()
	{
		++m_LightmapUpdateCount;
	}

	uint32_t LightingSettings::GetProbeVolumesUpdateCount() const
	{
		return m_ProbeVolumesUpdateCount;
	}

	void LightingSettings::IncrementProbeVolumesUpdateCount()
	{
		++m_ProbeVolumesUpdateCount;
	}

	uint32_t LightingSettings::GetReflectionProbesUpdateCount() const
	{
		return m_ReflectionProbesUpdateCount;
	}

	void LightingSettings::IncrementReflectionProbesUpdateCount()
	{
		++m_ReflectionProbesUpdateCount;
	}
}