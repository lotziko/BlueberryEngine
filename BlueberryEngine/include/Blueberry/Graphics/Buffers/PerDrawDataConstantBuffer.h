#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;
	class GfxBuffer;

	struct PerDrawData
	{
		Matrix modelMatrix;
		Matrix normalMatrix;
		Vector4 lightmapChartOffset;
	};

	class PerDrawDataConstantBuffer
	{
	public:
		static void BindData(const Matrix& localToWorldMatrix);
		static void BindDataInstanced(PerDrawData* data, size_t count);

	private:
		static GfxBuffer* s_StructuredBuffer;
		static GfxBuffer* s_ConstantBuffer;
		static GfxBuffer* s_ConstantBufferInstanced;
	};
}