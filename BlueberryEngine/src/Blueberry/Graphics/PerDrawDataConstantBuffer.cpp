#include "bbpch.h"
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
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		const Matrix& modelMatrix = GfxDevice::GetGPUMatrix(localToWorldMatrix);

		CONSTANTS constants =
		{
			modelMatrix
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perDrawDataId, s_ConstantBuffer);
	}

	void PerDrawConstantBuffer::BindDataInstanced(Matrix* localToWorldMatrices, const UINT& count)
	{
		static size_t perDrawDataId = TO_HASH("_PerDrawData");

		if (s_StructuredBuffer == nullptr)
		{
			GfxDevice::CreateStructuredBuffer(2048, sizeof(CONSTANTS), s_StructuredBuffer);
		}

		s_StructuredBuffer->SetData(reinterpret_cast<char*>(localToWorldMatrices), count);
		GfxDevice::SetGlobalStructuredBuffer(perDrawDataId, s_StructuredBuffer);

		/*static size_t perDrawDataInstancedId = TO_HASH("PerDrawDataInstanced");

		if (s_ConstantBufferInstanced == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 128, s_ConstantBufferInstanced);
		}

		s_ConstantBufferInstanced->SetData(reinterpret_cast<char*>(localToWorldMatrices), sizeof(CONSTANTS) * count);
		GfxDevice::SetGlobalConstantBuffer(perDrawDataInstancedId, s_ConstantBufferInstanced);*/
	}
}