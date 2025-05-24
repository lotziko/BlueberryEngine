#include "PostProcessingConstantBuffer.h"

#include "GfxDevice.h"
#include "GfxBuffer.h"

#include "..\Core\Time.h"

namespace Blueberry
{
	struct CONSTANTS
	{
		Vector4 exposureTime;
	};

	void PostProcessingConstantBuffer::BindData(const float& exposure)
	{
		static size_t postProcessingDataId = TO_HASH("_PostProcessingData");

		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(CONSTANTS) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		CONSTANTS constants =
		{
			Vector4(exposure, Time::GetFrameCount() / 60.0f / 10, 0, 0)
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalBuffer(postProcessingDataId, s_ConstantBuffer);
	}
}
