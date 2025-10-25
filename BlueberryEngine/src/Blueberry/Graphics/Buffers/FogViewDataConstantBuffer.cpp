#include "FogViewDataConstantBuffer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "..\RenderContext.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	struct FogViewData
	{
		Vector4 viewInvCount;
		Vector4 viewDX[2];
		Vector4 viewDY[2];
		Vector4 viewCorner[2];
		Vector4 viewPos[2];
		Matrix previousViewProj[2];
		Vector4 fogNearRange;
		Vector4 frustumVolumeInvSize;
		Vector4 frustumVolumeSize;
		//int4 _FogVolumesCount;
		//float4 _FogVolumeMin[MAX_FOG_VOLUMES];
		//float4 _FogVolumeScale[MAX_FOG_VOLUMES];
	};

	static size_t s_FogViewDataId = TO_HASH("FogViewData");

	void FogViewDataConstantBuffer::BindData(const CameraData& data, const Vector3Int& frustumVolumeSize)
	{
		static Matrix previousViewProj = Matrix::Identity;

		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.type = BufferType::Constant;
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(FogViewData) * 1;
			constantBufferProperties.isWritable = true;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		FogViewData constants = {};

		Camera* camera = data.camera;
		float fogNearClipPlane = camera->GetNearClipPlane();
		float fogFarClipPlane = 128;

		Matrix inverseView = camera->GetInverseViewMatrix();
		Matrix projection = Matrix::CreatePerspectiveFieldOfView(ToRadians(camera->GetFieldOfView()), camera->GetAspectRatio(), fogNearClipPlane, fogFarClipPlane);
		Transform* transform = camera->GetTransform();
		Vector4 position = transform->GetPosition();

		Frustum frustum = {};
		frustum.CreateFromMatrix(frustum, projection, false);
		frustum.Transform(frustum, inverseView);

		Vector3 corners[8];
		frustum.GetCorners(corners);

		Vector4 bottomLeft = corners[7] - position;
		Vector4 bottomRight = corners[6] - position;
		Vector4 topLeft = corners[4] - position;

		constants.viewInvCount = Vector4(1, 0, 0, 0);
		constants.viewDX[0] = bottomRight - bottomLeft;
		constants.viewDY[0] = topLeft - bottomLeft;
		constants.viewCorner[0] = bottomLeft;
		constants.viewPos[0] = position;
		constants.previousViewProj[0] = GfxDevice::GetGPUMatrix(previousViewProj);
		constants.fogNearRange = Vector4(1.0f - fogFarClipPlane / fogNearClipPlane, fogFarClipPlane / fogNearClipPlane, fogNearClipPlane / fogFarClipPlane, 1.0f / (1.0f - fogNearClipPlane / fogFarClipPlane));
		constants.frustumVolumeInvSize = Vector4(1.0f / frustumVolumeSize.x, 1.0f / frustumVolumeSize.y, 1.0f / frustumVolumeSize.z, 1.0f / (frustumVolumeSize.z - 1.0f));
		constants.frustumVolumeSize = Vector4(frustumVolumeSize.x, frustumVolumeSize.y, frustumVolumeSize.z, 0);
	
		previousViewProj = camera->GetViewProjectionMatrix();

		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(FogViewData));
		GfxDevice::SetGlobalBuffer(s_FogViewDataId, s_ConstantBuffer);
	}
}
