#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Texture2D;
	class Texture3D;
	class ProbeVolumeInfo;
	class TextureCube;

	class LightingSettings : public Object
	{
		OBJECT_DECLARATION(LightingSettings)

	public:
		LightingSettings() = default;
		virtual ~LightingSettings() = default;

		Texture2D* GetLightmap() const;
		void SetLightmap(Texture2D* lightmap);

		List<Vector4>& GetChartOffsetScale();
		void SetChartOffsetScale(const List<Vector4>& chartOffsetScale);

		const List<ObjectPtr<ProbeVolumeInfo>>& GetProbeVolumes() const;
		void SetProbeVolumes(const List<ObjectPtr<ProbeVolumeInfo>>& probeVolumes);

		const List<ObjectPtr<TextureCube>>& GetReflectionProbes() const;
		void SetReflectionProbes(const List<ObjectPtr<TextureCube>>& reflectionProbes);

		uint32_t GetLightmapUpdateCount() const;
		void IncrementLightmapUpdateCount();

		uint32_t GetProbeVolumesUpdateCount() const;
		void IncrementProbeVolumesUpdateCount();

		uint32_t GetReflectionProbesUpdateCount() const;
		void IncrementReflectionProbesUpdateCount();

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		List<ObjectPtr<ProbeVolumeInfo>> m_ProbeVolumes;
		List<ObjectPtr<TextureCube>> m_ReflectionProbes;
		List<Vector4> m_ChartOffsetScale;

		uint32_t m_LightmapUpdateCount = 0;
		uint32_t m_ProbeVolumesUpdateCount = 0;
		uint32_t m_ReflectionProbesUpdateCount = 0;
	};
}