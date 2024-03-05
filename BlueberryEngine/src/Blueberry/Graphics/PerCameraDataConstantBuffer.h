#pragma once

namespace Blueberry
{
	class BaseCamera;
	class GfxConstantBuffer;

	class PerCameraDataConstantBuffer
	{
	public:
		static void BindData(BaseCamera* camera);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}