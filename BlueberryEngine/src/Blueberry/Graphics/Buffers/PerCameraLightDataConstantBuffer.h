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
		static GfxBuffer* s_ConstantBuffer;
		static GfxBuffer* s_PointLightsBuffer;
		static GfxBuffer* s_SpotLightsBuffer;
		static GfxBuffer* s_ShadowsBuffer;
		static GfxBuffer* s_ReflectionProbesBuffer;
	};
}