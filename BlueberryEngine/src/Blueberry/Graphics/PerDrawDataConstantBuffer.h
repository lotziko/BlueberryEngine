#pragma once

namespace Blueberry
{
	class Camera;
	class GfxConstantBuffer;
	class GfxStructuredBuffer;

	class PerDrawConstantBuffer
	{
	public:
		static void BindData(const Matrix& localToWorldMatrix);
		static void BindDataInstanced(Matrix* localToWorldMatrices, const uint32_t& count);

	private:
		static inline GfxStructuredBuffer* s_StructuredBuffer = nullptr;
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
		static inline GfxConstantBuffer* s_ConstantBufferInstanced = nullptr;
	};
}