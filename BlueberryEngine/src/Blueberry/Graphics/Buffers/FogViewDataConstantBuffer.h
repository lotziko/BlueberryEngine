#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct CameraData;
	class GfxBuffer;

	class FogViewDataConstantBuffer
	{
	public:
		static void BindData(const CameraData& data, const Vector3Int& frustumVolumeSize);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
	};
}