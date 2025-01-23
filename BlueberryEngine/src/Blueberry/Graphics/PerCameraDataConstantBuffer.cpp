#include "bbpch.h"
#include "PerCameraDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Camera.h"
#include "GfxDevice.h"
#include "GfxBuffer.h"

namespace Blueberry
{
	#define MAX_VIEW_COUNT 2

	struct CONSTANTS
	{
		Vector4Int viewCount;
		Matrix viewMatrix[MAX_VIEW_COUNT];
		Matrix projectionMatrix[MAX_VIEW_COUNT];
		Matrix viewProjectionMatrix[MAX_VIEW_COUNT];
		Matrix inverseViewMatrix[MAX_VIEW_COUNT];
		Matrix inverseProjectionMatrix[MAX_VIEW_COUNT];
		Matrix inverseViewProjectionMatrix[MAX_VIEW_COUNT];
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

		const int viewCount = GfxDevice::GetViewCount();
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

		CONSTANTS constants = {};
		constants.viewCount = { viewCount, 0, 0, 0 };
		if (viewCount == 1)
		{
			constants.viewMatrix[0] = view;
			constants.projectionMatrix[0] = projection;
			constants.viewProjectionMatrix[0] = viewProjection;
			constants.inverseViewMatrix[0] = inverseView;
			constants.inverseProjectionMatrix[0] = inverseProjection;
			constants.inverseViewProjectionMatrix[0] = inverseViewProjection;
		}
		else
		{
			for (int i = 0; i < viewCount; ++i)
			{
				constants.viewMatrix[i] = view;
				constants.projectionMatrix[i] = projection;
				constants.viewProjectionMatrix[i] = GfxDevice::GetGPUMatrix(camera->GetViewProjectionMatrix());
				constants.inverseViewMatrix[i] = inverseView;
				constants.inverseProjectionMatrix[i] = inverseProjection;
				constants.inverseViewProjectionMatrix[i] = inverseViewProjection;
			}
		}
		constants.cameraPositionWS = position;
		constants.cameraForwardDirectionWS = direction;
		constants.cameraNearFarClipPlane = nearFar;
		constants.cameraSizeInvSize = sizeInvSize;

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
		constants.viewCount = { 1, 0, 0, 0 };
		constants.viewProjectionMatrix[0] = GfxDevice::GetGPUMatrix(viewProjection);

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(constants));
		GfxDevice::SetGlobalConstantBuffer(perCameraDataId, s_ConstantBuffer);
	}
}
