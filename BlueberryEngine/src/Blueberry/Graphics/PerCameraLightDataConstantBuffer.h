#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Transform;
	class Light;

	struct LightData
	{
		Transform* transform;
		Light* light;
	};

	class GfxConstantBuffer;

	class PerCameraLightDataConstantBuffer
	{
	public:
		static void BindData(const LightData& mainLight, const List<LightData>& lights, const Vector2Int& shadowAtlasSize);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}