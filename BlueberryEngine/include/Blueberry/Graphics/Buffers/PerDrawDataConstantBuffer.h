#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;
	class GfxBuffer;

	class PerDrawDataConstantBuffer
	{
	public:
		static void BindData(const Matrix& localToWorldMatrix);
		static void BindDataInstanced(std::pair<Matrix, Vector4>* data, const uint32_t& count);

	private:
		static GfxBuffer* s_StructuredBuffer;
		static GfxBuffer* s_ConstantBuffer;
		static GfxBuffer* s_ConstantBufferInstanced;
	};
}