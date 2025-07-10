#include "PostProcessingConstantBuffer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	struct PostProcessingData
	{
		Vector4 exposureTime;
	};

	static size_t s_PostProcessingDataId = TO_HASH("_PostProcessingData");

	void PostProcessingConstantBuffer::BindData(const float& exposure)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(PostProcessingData) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		PostProcessingData constants =
		{
			Vector4(exposure, Time::GetFrameCount() / 60.0f / 10, 0, 0)
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(PostProcessingData));
		GfxDevice::SetGlobalBuffer(s_PostProcessingDataId, s_ConstantBuffer);
	}
}
