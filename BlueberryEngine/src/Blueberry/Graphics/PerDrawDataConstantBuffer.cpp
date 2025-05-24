#include "PerDrawDataConstantBuffer.h"

#include "GfxDevice.h"
#include "GfxBuffer.h"

namespace Blueberry
{
	struct CONSTANTS
	{
		Matrix modelMatrix;
	};

	void PerDrawConstantBuffer::BindData(const Matrix& localToWorldMatrix)
	{
		static size_t perDrawDataId = TO_HASH("PerDrawData");

		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(CONSTANTS) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		const Matrix& modelMatrix = GfxDevice::GetGPUMatrix(localToWorldMatrix);

		CONSTANTS constants =
		{
			modelMatrix
		};

		s_ConstantBuffer->SetData(&constants, sizeof(constants));
		GfxDevice::SetGlobalBuffer(perDrawDataId, s_ConstantBuffer);
	}

	void PerDrawConstantBuffer::BindDataInstanced(Matrix* localToWorldMatrices, const uint32_t& count)
	{
		static size_t perDrawDataId = TO_HASH("_PerDrawData");

		if (s_StructuredBuffer == nullptr)
		{
			BufferProperties structuredBufferProperties = {};
			structuredBufferProperties.type = BufferType::Structured;
			structuredBufferProperties.elementCount = 2048;
			structuredBufferProperties.elementSize = sizeof(CONSTANTS);
			structuredBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(structuredBufferProperties, s_StructuredBuffer);
		}

		s_StructuredBuffer->SetData(localToWorldMatrices, count * sizeof(CONSTANTS));
		GfxDevice::SetGlobalBuffer(perDrawDataId, s_StructuredBuffer);
	}
}