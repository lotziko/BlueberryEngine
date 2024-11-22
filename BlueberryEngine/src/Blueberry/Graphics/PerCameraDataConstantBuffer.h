#pragma once

namespace Blueberry
{
	class Camera;
	class GfxConstantBuffer;

	class PerCameraDataConstantBuffer
	{
	public:
		static void BindData(Camera* camera);
		static void BindData(const Matrix& viewProjection);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}