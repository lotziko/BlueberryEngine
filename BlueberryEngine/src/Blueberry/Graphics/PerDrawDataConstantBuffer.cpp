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
}