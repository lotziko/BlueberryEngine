#include "RenderContext.h"

#include "..\Scene\Scene.h"
#include "..\Core\Time.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"

#include "..\Graphics\StandardMeshes.h"
#include "..\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "..\Graphics\GfxDevice.h"
#include "..\Graphics\GfxBuffer.h"
#include "..\Graphics\PerCameraDataConstantBuffer.h"
#include "..\Graphics\PerDrawDataConstantBuffer.h"
#include "..\Threading\JobSystem.h"

#include "..\Graphics\LightHelper.h"
#include "..\Graphics\RendererTree.h"

namespace Blueberry
{
	const uint32_t INSTANCE_BUFFER_SIZE = 2048;
	
	struct DrawingOperation
	{
		Matrix matrix;
		Mesh* mesh;
		ObjectId meshId;
		uint8_t submeshIndex;
		Material* material;
		ObjectId materialId;
		uint8_t instanceCount;
	};
	
	static List<DrawingOperation> s_DrawingOperations = {};
	static List<MeshRenderer*> s_MeshRenderers = {};
	static std::pair<Object*, uint8_t> s_LastCullerInfo = {};

	bool CompareOperations(const DrawingOperation& o1, const DrawingOperation& o2)
	{
		if (o1.materialId != o2.materialId)
		{
			return o1.materialId < o2.materialId;
		}
		if (o1.meshId != o2.meshId)
		{
			return o1.meshId < o2.meshId;
		}
		return o1.submeshIndex < o2.submeshIndex;
	}

	void BindOperationsRenderers()
	{
		uint32_t operationCount = static_cast<uint32_t>(s_DrawingOperations.size());
		Matrix matrices[INSTANCE_BUFFER_SIZE];
		for (uint32_t i = 0; i < operationCount; ++i)
		{
			matrices[i] = GfxDevice::GetGPUMatrix(s_DrawingOperations[i].matrix);
		}
		PerDrawConstantBuffer::BindDataInstanced(matrices, operationCount);
	}

	void GatherOperations(const CullingResults& results, Object* cullerObject, const uint8_t& index = 0)
	{
		if (s_LastCullerInfo.first == cullerObject && s_LastCullerInfo.second == index)
		{
			return;
		}
		else
		{
			s_LastCullerInfo = std::make_pair(cullerObject, index);
		}

		s_DrawingOperations.clear();

		uint32_t cullerIndex = 0;
		uint32_t cullerCount = static_cast<uint32_t>(results.cullerInfos.size());
		for (uint32_t i = 0; i < cullerCount; ++i)
		{
			auto& cullerInfo = results.cullerInfos[i];
			if (cullerInfo.object == cullerObject && cullerInfo.index == index)
			{
				cullerIndex = i;
				break;
			}
		}

		auto begin = results.cullerInfos[cullerIndex].renderers.begin();
		auto end = results.cullerInfos[cullerIndex].renderers.end();
		for (auto it = begin; it < end; ++it)
		{
			Renderer* renderer = static_cast<Renderer*>(ObjectDB::GetObject(*it));
			if (renderer->GetType() == MeshRenderer::Type)
			{
				auto meshRenderer = static_cast<MeshRenderer*>(renderer);
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh == nullptr)
				{
					continue;
				}
				Matrix matrix = meshRenderer->GetTransform()->GetLocalToWorldMatrix();
				uint32_t subMeshCount = mesh->GetSubMeshCount();
				if (subMeshCount > 1)
				{
					for (uint32_t i = 0; i < subMeshCount; ++i)
					{
						Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial(i));
						if (material != nullptr)
						{
							s_DrawingOperations.emplace_back(DrawingOperation{ matrix, mesh, mesh->GetObjectId(), static_cast<uint8_t>(i), material, material->GetObjectId(), 1 });
						}
					}
				}
				else
				{
					Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial());
					if (material != nullptr)
					{
						s_DrawingOperations.emplace_back(DrawingOperation{ matrix, mesh, mesh->GetObjectId(), 255, material, material->GetObjectId(), 1 });
					}
				}
			}
		}
		std::sort(s_DrawingOperations.begin(), s_DrawingOperations.end(), CompareOperations);

		BindOperationsRenderers();
	}

	void RenderContext::Cull(Scene* scene, CameraData& cameraData, CullingResults& results)
	{
		Camera* camera = cameraData.camera;
		if (scene == nullptr || camera == nullptr)
		{
			return;
		}

		results.camera = camera;
		results.lights.clear();
		results.cullerInfos.clear();

		Matrix view = cameraData.isMultiview ? cameraData.multiviewViewMatrix[0].Invert() : camera->GetInverseViewMatrix();
		Matrix projection = cameraData.isMultiview ? cameraData.multiviewProjectionMatrix[0] : camera->GetProjectionMatrix();
		Frustum cameraFrustum;
		cameraFrustum.CreateFromMatrix(cameraFrustum, projection, false);
		cameraFrustum.Transform(cameraFrustum, view);

		CullingResults::CullerInfo cameraFrustumInfo = {};
		cameraFrustumInfo.object = camera;
		cameraFrustum.GetPlanes(&cameraFrustumInfo.planes[0], &cameraFrustumInfo.planes[1], &cameraFrustumInfo.planes[2], &cameraFrustumInfo.planes[3], &cameraFrustumInfo.planes[4], &cameraFrustumInfo.planes[5]);
		results.cullerInfos.emplace_back(cameraFrustumInfo);

		if (s_LastCullingFrame < Time::GetFrameCount())
		{
			// TODO move to frame start or find the other away to react on the transform movement
			for (auto component : scene->GetIterator<MeshRenderer>())
			{
				auto meshRenderer = static_cast<MeshRenderer*>(component.second);
				meshRenderer->Update();
			}
			s_LastCullingFrame = Time::GetFrameCount();
		}

		for (auto component : scene->GetIterator<Light>())
		{
			auto light = static_cast<Light*>(component.second);
			auto transform = light->GetTransform();
			// TODO light types
			Sphere bounds = Sphere(transform->GetPosition(), light->GetRange());
			LightType type = light->GetType();
			if (type == LightType::Directional || bounds.ContainedBy(cameraFrustumInfo.planes[0], cameraFrustumInfo.planes[1], cameraFrustumInfo.planes[2], cameraFrustumInfo.planes[3], cameraFrustumInfo.planes[4], cameraFrustumInfo.planes[5]))
			{
				results.lights.emplace_back(light);

				CullingResults::CullerInfo lightCullerInfo = {};
				lightCullerInfo.object = light;
				
				if (light->IsCastingShadows())
				{
					if (light->GetType() == LightType::Directional)
					{
						float planes[] = { 0.01f, 5.0f, 15.0f, 43.0f };//{ 0.01f, 5.0f, 20.0f, 100.0f };

						Quaternion rotation = transform->GetRotation();
						Vector3 forward = Vector3::Transform(Vector3::Forward, rotation);
						Vector3 right = Vector3::Transform(Vector3::Right, rotation);
						Vector3 up = Vector3::Transform(Vector3::Up, rotation);

						for (int i = 0; i < 3; ++i)
						{
							float nearPlane = planes[i];
							float farPlane = planes[i + 1];

							Frustum cameraSliceFrustum;
							cameraSliceFrustum.CreateFromMatrix(cameraSliceFrustum, Matrix::CreatePerspectiveFieldOfView(ToRadians(camera->GetFieldOfView()), camera->GetAspectRatio(), nearPlane, farPlane), false);
							cameraSliceFrustum.Transform(cameraSliceFrustum, view);

							// Find near and far planes

							Vector3 corners[8];
							cameraSliceFrustum.GetCorners(corners);

							float minZ = FLT_MAX;
							float maxZ = FLT_MIN;

							for (int j = 0; j < 8; ++j)
							{
								Vector3 trf = Vector3::Transform(corners[j], view);
								minZ = std::min(minZ, trf.z);
								maxZ = std::max(maxZ, trf.z);
							}

							constexpr float zMult = 10.0f;
							if (minZ < 0)
							{
								minZ *= zMult;
							}
							else
							{
								minZ /= zMult;
							}
							if (maxZ < 0)
							{
								maxZ /= zMult;
							}
							else
							{
								maxZ *= zMult;
							}
							float zRange = maxZ - minZ;

							// Create frustum

							Sphere frustumSphere;
							Sphere::CreateFromFrustum(frustumSphere, cameraSliceFrustum);

							float radius = frustumSphere.Radius;
							Vector3 center = frustumSphere.Center;
							const float shadowSize = 1024.0f;

							// https://www.gamedev.net/forums/topic/497259-stable-cascaded-shadow-maps/

							Matrix projection = Matrix::CreateOrthographicOffCenter(-radius, radius, -radius, radius, 0.001f, zRange);
							Vector3 origin = center - forward * (zRange / 2.0f);
							Matrix view = Matrix::CreateLookAt(origin, center, up);
							Matrix viewProjection = view * projection;

							Vector3 centerLS = Vector3(0, 0, 0);
							centerLS = Vector3::Transform(centerLS, viewProjection);
							float texCoordX = centerLS.x * shadowSize * 0.5f;
							float texCoordY = centerLS.y * shadowSize * 0.5f;

							float texCoordRoundedX = std::round(texCoordX);
							float texCoordRoundedY = std::round(texCoordY);

							float dx = texCoordRoundedX - texCoordX;
							float dy = texCoordRoundedY - texCoordY;

							dx /= shadowSize * 0.5f;
							dy /= shadowSize * 0.5f;

							viewProjection *= Matrix::CreateTranslation(dx, dy, 0);

							light->m_WorldToShadow[i] = viewProjection;
							light->m_ShadowCascades[i] = Vector4(center.x, center.y, center.z, std::pow(radius, 2));

							GetOrthographicPlanes(viewProjection.Invert(), &lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);

							lightCullerInfo.index = i;
							results.cullerInfos.emplace_back(lightCullerInfo);
						}
					}
					else
					{
						Matrix view = LightHelper::GetViewMatrix(light);
						Matrix projection = LightHelper::GetProjectionMatrix(light);

						Frustum lightFrustum;
						lightFrustum.CreateFromMatrix(lightFrustum, projection, false);
						lightFrustum.Transform(lightFrustum, transform->GetLocalToWorldMatrix());

						lightFrustum.GetPlanes(&lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);
						light->m_WorldToShadow[0] = view * projection;

						results.cullerInfos.emplace_back(lightCullerInfo);
					}
				}
			}
		}

		JobSystem::Dispatch(static_cast<uint32_t>(results.cullerInfos.size()), 1, [&results, scene](JobDispatchArgs args)
		{
			scene->GetRendererTree().Cull(results.cullerInfos[args.jobIndex].planes, results.cullerInfos[args.jobIndex].renderers);
		});
		JobSystem::Wait();

		s_LastCullerInfo = std::make_pair(nullptr, 0);
	}

	void RenderContext::BindCamera(CullingResults& results, CameraData& cameraData)
	{
		Camera* camera = results.camera;
		PerCameraDataConstantBuffer::BindData(cameraData);
		GatherOperations(results, camera);
	}

	void RenderContext::DrawSky(Scene* scene)
	{
		for (auto component : scene->GetIterator<SkyRenderer>())
		{
			Material* material = static_cast<SkyRenderer*>(component.second)->GetMaterial();
			if (material != nullptr)
			{
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetCube(), material, 0));
			}
		}
	}

	void RenderContext::DrawShadows(CullingResults& results, ShadowDrawingSettings& shadowDrawingSettings)
	{
		Light* light = shadowDrawingSettings.light;

		PerCameraDataConstantBuffer::BindData(light->m_WorldToShadow[shadowDrawingSettings.sliceIndex]);
		GatherOperations(results, light, shadowDrawingSettings.sliceIndex);

		DrawingSettings drawingSettings = {};
		drawingSettings.passIndex = 2; // Shadow caster
		DrawRenderers(results, drawingSettings);
	}

	void RenderContext::DrawRenderers(CullingResults& results, DrawingSettings& drawingSettings)
	{
		uint8_t passIndex = drawingSettings.passIndex;

		if (s_IndexBuffer == nullptr)
		{
			GfxDevice::CreateVertexBuffer(INSTANCE_BUFFER_SIZE, sizeof(uint32_t), s_IndexBuffer);
			uint32_t indices[INSTANCE_BUFFER_SIZE];
			for (uint32_t i = 0; i < INSTANCE_BUFFER_SIZE; ++i)
			{
				indices[i] = i;
			}
			s_IndexBuffer->SetData(reinterpret_cast<float*>(indices), INSTANCE_BUFFER_SIZE);
		}

		// Draw meshes
		uint32_t operationCount = static_cast<uint32_t>(s_DrawingOperations.size());
		for (uint32_t i = 0; i < operationCount;)
		{
			auto& operation = s_DrawingOperations[i];
			if (operation.submeshIndex == 255)
			{
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, passIndex, s_IndexBuffer, i, operation.instanceCount));
			}
			else
			{
				auto& subMesh = operation.mesh->GetSubMesh(operation.submeshIndex);
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, subMesh.GetIndexCount(), subMesh.GetIndexStart(), operation.mesh->GetVertexCount(), passIndex, s_IndexBuffer, i, operation.instanceCount));
			}
			i += operation.instanceCount;
		}
	}
}
