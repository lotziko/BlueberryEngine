#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class GfxTexture;
	class GfxBuffer;
	class ComputeShader;
	class Camera;
	class PerCameraData;

	struct PerCameraExposureData
	{
		float recalculateTimer = 0.0f;
		float targetExposure = 0.15f;
		float currentExposure = 0.15f;
	};

	class AutoExposure
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Calculate(Camera* camera, GfxTexture* color, const Rectangle& viewport, PerCameraData& perCameraData);
		static float GetExposure(const PerCameraData& perCameraData);

	private:
		static ComputeShader* s_ExposureShader;
		static GfxBuffer* s_ExposureData;
		static GfxBuffer* s_Histogram;
		static GfxBuffer* s_Result;
	};
}