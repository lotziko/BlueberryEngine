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
		static inline GfxBuffer* s_StructuredBuffer = nullptr;
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
		static inline GfxBuffer* s_ConstantBufferInstanced = nullptr;
	};
}