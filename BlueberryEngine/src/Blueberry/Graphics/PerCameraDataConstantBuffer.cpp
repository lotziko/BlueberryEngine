#include "bbpch.h"
#include "PerCameraDataConstantBuffer.h"

#include "BaseCamera.h"
#include "GfxDevice.h"
#include "GfxBuffer.h"

namespace Blueberry
{
	struct CONSTANTS
	{
		Matrix viewMatrix;
		Matrix projectionMatrix;
		Matrix viewProjectionMatrix;
		Matrix inverseViewMatrix;
		Matrix inverseProjectionMatrix;
		Matrix inverseViewProjectionMatrix;
		Vector4 cameraPositionWS;
	};

	void PerCameraDataConstantBuffer::BindData(BaseCamera* camera)
	{
		static size_t perCameraDataId = TO_HASH("PerCameraData");

		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		const Matrix& view = GfxDevice::GetGPUMatrix(camera->GetViewMatrix());
		const Matrix& projection = GfxDevice::GetGPUMatrix(camera->GetProjectionMatrix());
		const Matrix& viewProjection = GfxDevice::GetGPUMatrix(camera->GetViewProjectionMatrix());
		const Matrix& inverseView = GfxDevice::GetGPUMatrix(camera->GetInverseViewMatrix());
		const Matrix& inverseProjection = GfxDevice::GetGPUMatrix(camera->GetInverseProjectionMatrix());
		const Matrix& inverseViewProjection = GfxDevice::GetGPUMatrix(camera->GetInverseViewProjectionMatrix());
		const Vector4& position = Vector4(camera->GetPosition());

		CONSTANTS constants =
		{
			view,
			projection,
			viewProjection,
			inverseView,
			inverseProjection,
			inverseViewProjection,
			position
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraDataId, s_ConstantBuffer);
	}
}
