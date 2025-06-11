#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer;
	class Light;

	class FogLightDataConstantBuffer
	{
	public:
		static void BindData(const List<Light*>& lights, const Vector2Int& shadowAtlasSize);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
	};
}