#include "bbpch.h"
#include "PerCameraDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Camera.h"
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
		Vector4 cameraForwardDirectionWS;
		Vector4 cameraNearFarClipPlane;
	};

	void PerCameraDataConstantBuffer::BindData(Camera* camera)
	{
		static size_t perCameraDataId = TO_HASH("PerCameraData");

		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		Transform* transform = camera->GetTransform();

		const Matrix& view = GfxDevice::GetGPUMatrix(camera->GetViewMatrix());
		const Matrix& projection = GfxDevice::GetGPUMatrix(camera->GetProjectionMatrix());
		const Matrix& viewProjection = GfxDevice::GetGPUMatrix(camera->GetViewProjectionMatrix());
		const Matrix& inverseView = GfxDevice::GetGPUMatrix(camera->GetInverseViewMatrix());
		const Matrix& inverseProjection = GfxDevice::GetGPUMatrix(camera->GetInverseProjectionMatrix());
		const Matrix& inverseViewProjection = GfxDevice::GetGPUMatrix(camera->GetInverseViewProjectionMatrix());
		const Vector4& position = Vector4(transform->GetPosition());
		const Vector4& direction = Vector4(Vector3::Transform(Vector3::Forward, transform->GetRotation()));
		const Vector4& params = Vector4(camera->GetNearClipPlane(), camera->GetFarClipPlane(), 0, 0);

		CONSTANTS constants =
		{
			view,
			projection,
			viewProjection,
			inverseView,
			inverseProjection,
			inverseViewProjection,
			position,
			direction,
			params
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraDataId, s_ConstantBuffer);
	}
}
