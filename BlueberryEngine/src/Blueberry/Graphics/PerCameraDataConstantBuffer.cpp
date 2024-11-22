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
		Vector4 cameraSizeInvSize;
	};

	static size_t perCameraDataId = TO_HASH("PerCameraData");

	void PerCameraDataConstantBuffer::BindData(Camera* camera)
	{
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
		const Vector4& nearFar = Vector4(camera->GetNearClipPlane(), camera->GetFarClipPlane(), 0, 0);
		const Vector2& pixelSize = camera->GetPixelSize();
		const Vector4& sizeInvSize = Vector4(pixelSize.x, pixelSize.y, 1.0f / pixelSize.x, 1.0f / pixelSize.y);

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
			nearFar,
			sizeInvSize
		};

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraDataId, s_ConstantBuffer);
	}

	void PerCameraDataConstantBuffer::BindData(const Matrix& viewProjection)
	{
		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		CONSTANTS constants = {};
		constants.viewProjectionMatrix = GfxDevice::GetGPUMatrix(viewProjection);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraDataId, s_ConstantBuffer);
	}
}
