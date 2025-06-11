#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Camera;
	class GfxTexture;
	class GfxBuffer;
	struct CameraData;

	class PerCameraDataConstantBuffer
	{
	public:
		static void BindData(Camera* camera, GfxTexture* target);
		static void BindData(CameraData& cameraData);
		static void BindData(const Matrix& viewProjection);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
	};
}