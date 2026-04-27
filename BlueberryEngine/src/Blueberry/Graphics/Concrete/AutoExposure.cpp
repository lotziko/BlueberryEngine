#include "AutoExposure.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	ComputeShader* AutoExposure::s_ExposureShader = nullptr;
	GfxBuffer* AutoExposure::s_ExposureData = nullptr;
	GfxBuffer* AutoExposure::s_Histogram = nullptr;
	GfxBuffer* AutoExposure::s_Result = nullptr;
	Dictionary<ObjectId, PerCameraExposureData> AutoExposure::s_PerCameraData = {};

	struct ExposureData
	{
		float minLogLuminance;
		float logLuminanceRange;
		float inverseLogLuminanceRange;
		float numPixels;
		Vector2 viewport;
		Vector2 dummy;
	};

	static size_t s_ScreenColorTextureId = TO_HASH("_ScreenColorTexture");
	static size_t s_ExposureDataId = TO_HASH("ExposureData");
	static size_t s_HistogramId = TO_HASH("_Histogram");
	static size_t s_ResultId = TO_HASH("_Result");

	void AutoExposure::Initialize()
	{
		s_ExposureShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/Exposure.compute"));

		BufferProperties constantBufferProperties = {};
		constantBufferProperties.elementCount = 1;
		constantBufferProperties.elementSize = sizeof(ExposureData);
		constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;
		GfxDevice::CreateBuffer(constantBufferProperties, s_ExposureData);

		BufferProperties histogramBufferProperties = {};
		histogramBufferProperties.elementCount = 256;
		histogramBufferProperties.elementSize = sizeof(uint32_t);
		histogramBufferProperties.format = BufferFormat::R32_UInt;
		histogramBufferProperties.usageFlags = BufferUsageFlags::UnorderedAccess;
		GfxDevice::CreateBuffer(histogramBufferProperties, s_Histogram);

		BufferProperties resultBufferProperties = {};
		resultBufferProperties.elementCount = 1;
		resultBufferProperties.elementSize = sizeof(float);
		resultBufferProperties.format = BufferFormat::R32_Float;
		resultBufferProperties.usageFlags = BufferUsageFlags::UnorderedAccess | BufferUsageFlags::CPUReadable;
		GfxDevice::CreateBuffer(resultBufferProperties, s_Result);
	}

	void AutoExposure::Shutdown()
	{
		delete s_ExposureData;
		delete s_Histogram;
		delete s_Result;
	}

	void AutoExposure::Calculate(Camera* camera, GfxTexture* color, const Rectangle& viewport)
	{
		PerCameraExposureData& perCameraData = s_PerCameraData[camera->GetObjectId()];
		
		if (perCameraData.recalculateTimer == 0)
		{
			perCameraData.recalculateTimer = 60;
			float minLogLum = -8.0f / 2;
			float maxLogLum = 3.5f / 2;

			ExposureData constants = {};
			constants.minLogLuminance = minLogLum;
			constants.logLuminanceRange = maxLogLum - minLogLum;
			constants.inverseLogLuminanceRange = 1.0f / (maxLogLum - minLogLum);
			constants.numPixels = static_cast<float>(viewport.width * viewport.height);
			constants.viewport = Vector2(static_cast<float>(viewport.width), static_cast<float>(viewport.height));

			s_ExposureData->SetData(reinterpret_cast<char*>(&constants), sizeof(ExposureData));

			GfxDevice::SetGlobalTexture(s_ScreenColorTextureId, color);
			GfxDevice::SetGlobalBuffer(s_ExposureDataId, s_ExposureData);
			GfxDevice::SetGlobalBuffer(s_HistogramId, s_Histogram);
			GfxDevice::SetGlobalBuffer(s_ResultId, s_Result);

			uint32_t groupsX = (static_cast<uint32_t>(viewport.width + 15)) / 16;
			uint32_t groupsY = (static_cast<uint32_t>(viewport.height + 15)) / 16;
			GfxDevice::Dispatch(s_ExposureShader->GetKernel(0), groupsX, groupsY, 1);
			GfxDevice::Dispatch(s_ExposureShader->GetKernel(1), 1, 1, 1);
			s_Result->GetData(&perCameraData.targetExposure);
		}
		else
		{
			--perCameraData.recalculateTimer;
		}
		float adaptationSpeed = 1.0f;
		float deltaTime = 1.0f / 60.0f;
		float t = 1.0f - std::expf(-adaptationSpeed * deltaTime);
		perCameraData.currentExposure = Math::Lerp(perCameraData.currentExposure, perCameraData.targetExposure, t);
	}

	float AutoExposure::GetExposure(Camera* camera)
	{
		PerCameraExposureData& perCameraData = s_PerCameraData[camera->GetObjectId()];
		return perCameraData.currentExposure;
	}
}
