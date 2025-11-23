#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Transform;
	class Camera;
	class Light;
	class SkyRenderer;
	class ProbeVolume;
	class ReflectionProbe;

	class GfxBuffer;

	class PerCameraLightDataConstantBuffer
	{
	public:
		static void BindData(Camera* camera, Light* mainLight, SkyRenderer* skyRenderer, ProbeVolume* probeVolume, const List<Light*>& lights, const List<ReflectionProbe*>& reflectionProbes, const Vector2Int& shadowAtlasSize);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
		static inline GfxBuffer* s_PointLightsBuffer = nullptr;
		static inline GfxBuffer* s_SpotLightsBuffer = nullptr;
		static inline GfxBuffer* s_ReflectionProbesBuffer = nullptr;
	};
}