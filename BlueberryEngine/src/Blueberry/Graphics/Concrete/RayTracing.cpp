#include "RayTracing.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxTopLevelAccelerationStructure.h"
#include "Blueberry\Graphics\RayTracingShader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	RayTracingShader* RayTracing::s_Shader = nullptr;
	GfxBuffer* RayTracing::s_ConstantBuffer = nullptr;
	GfxTopLevelAccelerationStructure* RayTracing::s_AccelerationStructure = nullptr;

	struct PerCameraData
	{
		Matrix inverseViewProjectionMatrix;
		Vector4 cameraPositionWS;
	};

	static size_t s_PerCameraDataId = TO_HASH("RTPerCameraData");
	static size_t s_OutputTextureId = TO_HASH("_OutputTexture");

	void RayTracing::Initialize()
	{
		s_Shader = static_cast<RayTracingShader*>(AssetLoader::Load("assets/shaders/Reflections.raytrace"));

		BufferProperties constantBufferProperties = {};
		constantBufferProperties.elementCount = 1;
		constantBufferProperties.elementSize = sizeof(PerCameraData) * 1;
		constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		GfxDevice::CreateTopLevelAccelerationStructure(s_AccelerationStructure);
	}

	void RayTracing::Shutdown()
	{
		Object::Destroy(s_Shader);
		delete s_ConstantBuffer;
		if (s_AccelerationStructure != nullptr)
		{
			delete s_AccelerationStructure;
		}
	}

	void RayTracing::Draw(Scene* scene, Camera* camera, GfxTexture* output, Rectangle viewport, Vector2Int size)
	{
		if (s_AccelerationStructure == nullptr)
		{
			return;
		}

		s_AccelerationStructure->Clear();
		for (auto& component : scene->GetIterator<MeshRenderer>())
		{
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
			s_AccelerationStructure->Add(meshRenderer->GetAccelerationStructure(), meshRenderer->GetMaterials(), meshRenderer->GetTransform()->GetLocalToWorldMatrix());
		}

		// TODO move miss into skybox material and make GfxTopLevelAccelerationStructure::Add for it
		Texture* skyboxTexture = nullptr;
		for (auto& component : scene->GetIterator<SkyRenderer>())
		{
			SkyRenderer* skyRenderer = static_cast<SkyRenderer*>(component.second);
			Material* material = skyRenderer->GetMaterial();
			if (material != nullptr)
			{
				Texture* baseMap = material->GetTexture(TO_HASH("_BaseMap"));
				if (baseMap != nullptr)
				{
					skyboxTexture = baseMap;
					break;
				}
			}
		}
		if (skyboxTexture == nullptr)
		{
			skyboxTexture = DefaultTextures::GetBlackCube();
		}
		GfxDevice::SetGlobalTexture(TO_HASH("_SkyboxTexture"), skyboxTexture->Get());

		PerCameraData constants = {};
		constants.inverseViewProjectionMatrix = GfxDevice::GetGPUMatrix(camera->GetInverseViewProjectionMatrix());
		constants.cameraPositionWS = Vector4(camera->GetTransform()->GetPosition());

		GfxDevice::SetGlobalTexture(s_OutputTextureId, output);
		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(PerCameraData));
		GfxDevice::SetGlobalBuffer(s_PerCameraDataId, s_ConstantBuffer);
		GfxDevice::DispatchRays(s_Shader, s_AccelerationStructure, viewport.width, viewport.height, 1);
	}
}
