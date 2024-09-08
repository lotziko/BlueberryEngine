#pragma once

namespace Blueberry
{
	class Camera;
	class GfxConstantBuffer;

	class PerDrawConstantBuffer
	{
	public:
		static void BindData(const Matrix& localToWorldMatrix);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}