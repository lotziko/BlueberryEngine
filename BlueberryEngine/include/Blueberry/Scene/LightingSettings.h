#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Texture2D;
	class Texture3D;
	class TextureCubeArray;

	class LightingSettings : public Object
	{
		OBJECT_DECLARATION(LightingSettings)

	public:
		LightingSettings() = default;
		virtual ~LightingSettings() = default;

		Texture2D* GetLightmap();
		void SetLightmap(Texture2D* lightmap);

		Texture3D* GetProbeVolume();
		void SetProbeVolume(Texture3D* probeVolume);

		TextureCubeArray* GetReflectionProbes();
		void SetReflectionProbes(TextureCubeArray* reflectionProbes);

		List<Vector4>& GetChartOffsetScale();
		void SetChartOffsetScale(const List<Vector4>& chartOffsetScale);

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		ObjectPtr<Texture3D> m_ProbeVolume;
		ObjectPtr<TextureCubeArray> m_ReflectionProbes;
		List<Vector4> m_ChartOffsetScale;
	};
}