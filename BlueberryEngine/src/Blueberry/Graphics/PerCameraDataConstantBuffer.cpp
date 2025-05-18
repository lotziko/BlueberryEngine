#include "PerCameraDataConstantBuffer.h"

#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "..\Graphics\RenderContext.h"
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
		CameraData cameraData = { camera };
		BindData(cameraData);
	}

	void PerCameraDataConstantBuffer::BindData(CameraData& cameraData)
	{
		if (s_ConstantBuffer == nullptr)
		{
			GfxDevice::CreateConstantBuffer(sizeof(CONSTANTS) * 1, s_ConstantBuffer);
		}

		Camera* camera = cameraData.camera;
		Transform* transform = camera->GetTransform();

		const int viewCount = GfxDevice::GetViewCount();
		CONSTANTS constants = {};
		constants.viewCount = { viewCount, 0, 0, 0 };

		const Vector4& position = Vector4(transform->GetPosition());
		const Vector4& direction = Vector4(Vector3::Transform(Vector3::Forward, transform->GetRotation()));
		const Vector4& nearFar = Vector4(camera->GetNearClipPlane(), camera->GetFarClipPlane(), 0, 0);
		
		if (viewCount == 1)
		{
			const Matrix& view = GfxDevice::GetGPUMatrix(camera->GetViewMatrix());
			const Matrix& projection = GfxDevice::GetGPUMatrix(camera->GetProjectionMatrix());
			const Matrix& viewProjection = GfxDevice::GetGPUMatrix(camera->GetViewProjectionMatrix());
			const Matrix& inverseView = GfxDevice::GetGPUMatrix(camera->GetInverseViewMatrix());
			const Matrix& inverseProjection = GfxDevice::GetGPUMatrix(camera->GetInverseProjectionMatrix());
			const Matrix& inverseViewProjection = GfxDevice::GetGPUMatrix(camera->GetInverseViewProjectionMatrix());
			const Vector2& pixelSize = Vector2(static_cast<float>(cameraData.size.x), static_cast<float>(cameraData.size.y));
			const Vector4& sizeInvSize = Vector4(pixelSize.x, pixelSize.y, 1.0f / pixelSize.x, 1.0f / pixelSize.y);

			constants.viewMatrix[0] = view;
			constants.projectionMatrix[0] = projection;
			constants.viewProjectionMatrix[0] = viewProjection;
			constants.inverseViewMatrix[0] = inverseView;
			constants.inverseProjectionMatrix[0] = inverseProjection;
			constants.inverseViewProjectionMatrix[0] = inverseViewProjection;
			constants.cameraSizeInvSize = sizeInvSize;
		}
		else
		{
			for (int i = 0; i < viewCount; ++i)
			{
				const Matrix& view = GfxDevice::GetGPUMatrix(cameraData.multiviewViewMatrix[i]);
				const Matrix& projection = GfxDevice::GetGPUMatrix(cameraData.multiviewProjectionMatrix[i]);
				const Matrix& viewProjection = GfxDevice::GetGPUMatrix(cameraData.multiviewViewMatrix[i] * cameraData.multiviewProjectionMatrix[i]);
				const Matrix& inverseView = GfxDevice::GetGPUMatrix(cameraData.multiviewViewMatrix[i].Invert());
				const Matrix& inverseProjection = GfxDevice::GetGPUMatrix(cameraData.multiviewProjectionMatrix[i].Invert());
				const Matrix& inverseViewProjection = GfxDevice::GetGPUMatrix((cameraData.multiviewViewMatrix[i] * cameraData.multiviewProjectionMatrix[i]).Invert());
				const Vector2& pixelSize = Vector2(static_cast<float>(cameraData.multiviewViewport.width), static_cast<float>(cameraData.multiviewViewport.height));
				const Vector4& sizeInvSize = Vector4(pixelSize.x, pixelSize.y, 1.0f / pixelSize.x, 1.0f / pixelSize.y);

				constants.viewMatrix[i] = view;
				constants.projectionMatrix[i] = projection;
				constants.viewProjectionMatrix[i] = viewProjection;
				constants.inverseViewMatrix[i] = inverseView;
				constants.inverseProjectionMatrix[i] = inverseProjection;
				constants.inverseViewProjectionMatrix[i] = inverseViewProjection;
				constants.cameraSizeInvSize = sizeInvSize;
			}
		}
		constants.cameraPositionWS = position;
		constants.cameraForwardDirectionWS = direction;
		constants.cameraNearFarClipPlane = nearFar;

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
