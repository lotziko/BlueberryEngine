#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Transform;
	class Light;
	class SkyRenderer;

	class GfxBuffer;

	class PerCameraLightDataConstantBuffer
	{
	public:
		static void BindData(Light* mainLight, SkyRenderer* skyRenderer, const List<Light*>& lights, const Vector2Int& shadowAtlasSize);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
	};
}