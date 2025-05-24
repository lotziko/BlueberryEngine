#include "AutoExposure.h"

#include "..\Assets\AssetLoader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "GfxBuffer.h"
#include "GfxDevice.h"
#include "GfxTexture.h"

namespace Blueberry
{
	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ExposureDataId = TO_HASH("_ExposureData");
	static size_t s_HistogramId = TO_HASH("_Histogram");
	static size_t s_ResultId = TO_HASH("_Result");

	float Lerp(float a, float b, float t)
	{
		return a + (b - a) * t;
	}

	float AutoExposure::Calculate(GfxTexture* color, const Rectangle& viewport)
	{
		if (s_ExposureShader == nullptr)
		{
			s_ExposureShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/Exposure.compute"));

			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(float) * 8;
			constantBufferProperties.isWritable = true;
			GfxDevice::CreateBuffer(constantBufferProperties, s_ExposureData);

			BufferProperties histogramBufferProperties = {};
			histogramBufferProperties.type = BufferType::Raw;
			histogramBufferProperties.elementCount = 256;
			histogramBufferProperties.elementSize = sizeof(uint32_t);
			histogramBufferProperties.isUnorderedAccess = true;
			histogramBufferProperties.format = BufferFormat::R32_UInt;
			GfxDevice::CreateBuffer(histogramBufferProperties, s_Histogram);

			BufferProperties resultBufferProperties = {};
			resultBufferProperties.type = BufferType::Raw;
			resultBufferProperties.elementCount = 1;
			resultBufferProperties.elementSize = sizeof(float);
			resultBufferProperties.isReadable = true;
			resultBufferProperties.isUnorderedAccess = true;
			resultBufferProperties.format = BufferFormat::R32_Float;
			GfxDevice::CreateBuffer(resultBufferProperties, s_Result);
		}

		if (s_RecalculateTimer == 0)
		{
			float minLogLum = -8.0f;
			float maxLogLum = 3.5f;
			float histogramParams[8] = { minLogLum, maxLogLum - minLogLum, 1.0f / (maxLogLum - minLogLum), static_cast<float>(viewport.width * viewport.height), static_cast<float>(viewport.width), static_cast<float>(viewport.height), 0, 0 };
			s_ExposureData->SetData(reinterpret_cast<char*>(histogramParams), sizeof(float) * 8);

			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, color);
			GfxDevice::SetGlobalBuffer(s_ExposureDataId, s_ExposureData);
			GfxDevice::SetGlobalBuffer(s_HistogramId, s_Histogram);
			GfxDevice::SetGlobalBuffer(s_ResultId, s_Result);

			uint32_t groupsX = (static_cast<uint32_t>(viewport.width + 15)) / 16;
			uint32_t groupsY = (static_cast<uint32_t>(viewport.height + 15)) / 16;
			GfxDevice::Dispatch(s_ExposureShader->GetKernel(0), groupsX, groupsY, 1);
			GfxDevice::Dispatch(s_ExposureShader->GetKernel(1), 1, 1, 1);
			s_Result->GetData(&s_TargetExposure);
		}
		float adaptationSpeed = 1.0f;
		float deltaTime = 1.0f / 60.0f;
		float t = 1.0f - std::expf(-adaptationSpeed * deltaTime);
		s_CurrentExposure = Lerp(s_CurrentExposure, s_TargetExposure, t);
		return s_CurrentExposure;
	}
}
