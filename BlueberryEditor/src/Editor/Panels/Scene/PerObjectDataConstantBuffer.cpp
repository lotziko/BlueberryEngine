#include "PerObjectDataConstantBuffer.h"

#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	GfxBuffer* PerObjectDataConstantBuffer::s_ConstantBuffer = nullptr;

	static size_t s_PerObjectDataId = TO_HASH("PerObjectData");

	struct PerObjectData
	{
		Color objectId;
	};

	void PerObjectDataConstantBuffer::BindData(const Color & indexColor)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(CONSTANTS) * 1;
			constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		PerObjectData constants = {};
		constants.objectId = indexColor;

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(PerObjectData));
		GfxDevice::SetGlobalBuffer(s_PerObjectDataId, s_ConstantBuffer);
	}
}