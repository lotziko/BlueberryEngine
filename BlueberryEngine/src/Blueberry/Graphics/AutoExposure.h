#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxTexture;
	class GfxBuffer;
	class ComputeShader;

	class AutoExposure
	{
	public:
		static float Calculate(GfxTexture* color, const Rectangle& viewport);

	private:
		static inline ComputeShader* s_ExposureShader = nullptr;
		static inline GfxBuffer* s_ExposureData = nullptr;
		static inline GfxBuffer* s_Histogram = nullptr;
		static inline GfxBuffer* s_Result = nullptr;
		static inline uint32_t s_RecalculateTimer = 0;
		static inline float s_TargetExposure = 0.15f;
		static inline float s_CurrentExposure = 0.15f;
	};
}