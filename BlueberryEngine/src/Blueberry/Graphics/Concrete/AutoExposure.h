#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class GfxTexture;
	class GfxBuffer;
	class ComputeShader;
	class Camera;

	struct PerCameraExposureData
	{
		uint32_t recalculateTimer = 0;
		float targetExposure = 0.15f;
		float currentExposure = 0.15f;
	};

	class AutoExposure
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Calculate(Camera* camera, GfxTexture* color, const Rectangle& viewport);
		static float GetExposure(Camera* camera);

	private:
		static ComputeShader* s_ExposureShader;
		static GfxBuffer* s_ExposureData;
		static GfxBuffer* s_Histogram;
		static GfxBuffer* s_Result;
		static Dictionary<ObjectId, PerCameraExposureData> s_PerCameraData;
	};
}