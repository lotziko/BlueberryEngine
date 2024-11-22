#include "bbpch.h"
#include "RenderContext.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"

#include "Blueberry\Graphics\LightHelper.h"

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
	
	static std::vector<DrawingOperation> s_DrawingOperations = {};
	static std::pair<Object*, uint8_t> s_LastCullerInfo = {};

	bool CompareOperations(DrawingOperation o1, DrawingOperation o2)
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
		uint32_t operationCount = s_DrawingOperations.size();
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

		uint32_t frustumMask;
		for (int i = 0; i < results.cullerInfos.size(); ++i)
		{
			auto& cullerInfo = results.cullerInfos[i];
			if (cullerInfo.object == cullerObject && cullerInfo.index == index)
			{
				frustumMask = 1 << i;
				break;
			}
		}

		for (auto ptr = results.rendererInfos.begin(); ptr < results.rendererInfos.end(); ++ptr)
		{
			CullingResults::RendererInfo info = *ptr;
			if (info.frustumMask & frustumMask)
			{
				if (info.type == MeshRenderer::Type)
				{
					auto meshRenderer = static_cast<MeshRenderer*>(info.renderer);
					Mesh* mesh = meshRenderer->GetMesh();
					Matrix matrix = meshRenderer->GetTransform()->GetLocalToWorldMatrix();
					uint32_t subMeshCount = mesh->GetSubMeshCount();
					if (subMeshCount > 1)
					{
						for (uint32_t i = 0; i < subMeshCount; ++i)
						{
							Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial(i));
							if (material != nullptr)
							{
								SubMeshData* subMesh = mesh->GetSubMesh(i);
								s_DrawingOperations.emplace_back(DrawingOperation{ matrix, mesh, mesh->GetObjectId(), (uint8_t)i, material, material->GetObjectId(), 1 });
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
		}
		std::sort(s_DrawingOperations.begin(), s_DrawingOperations.end(), CompareOperations);

		BindOperationsRenderers();
	}

	void RenderContext::Cull(Scene* scene, Camera* camera, CullingResults& results)
	{
		if (scene == nullptr || camera == nullptr)
		{
			return;
		}

		results.camera = camera;
		results.lights.clear();
		results.rendererInfos.clear();
		results.cullerInfos.clear();

		Matrix view = camera->GetInverseViewMatrix();
		Matrix projection = camera->GetProjectionMatrix();
		Frustum cameraFrustum;
		cameraFrustum.CreateFromMatrix(cameraFrustum, projection, false);
		cameraFrustum.Transform(cameraFrustum, view);

		std::vector<CullingResults::CullerInfo> cullerInfos;
		CullingResults::CullerInfo cameraFrustumInfo = {};
		cameraFrustumInfo.object = camera;
		cameraFrustum.GetPlanes(&cameraFrustumInfo.nearPlane, &cameraFrustumInfo.farPlane, &cameraFrustumInfo.rightPlane, &cameraFrustumInfo.leftPlane, &cameraFrustumInfo.topPlane, &cameraFrustumInfo.bottomPlane);
		cullerInfos.emplace_back(cameraFrustumInfo);

		for (auto component : scene->GetIterator<Light>())
		{
			auto light = static_cast<Light*>(component.second);
			auto transform = light->GetTransform();
			// TODO light types
			Sphere bounds = Sphere(transform->GetPosition(), light->GetRange());
			LightType type = light->GetType();
			if (type == LightType::Directional || bounds.ContainedBy(cameraFrustumInfo.nearPlane, cameraFrustumInfo.farPlane, cameraFrustumInfo.rightPlane, cameraFrustumInfo.leftPlane, cameraFrustumInfo.topPlane, cameraFrustumInfo.bottomPlane))
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

							Vector3 corners[8];
							cameraSliceFrustum.GetCorners(corners);

							Vector3 center = Vector3::Zero;
							for (int j = 0; j < 8; ++j)
							{
								center += corners[j];
							}
							center /= 8;

							Matrix view = Matrix::CreateLookAt(center, center + forward, up);

							Vector3 min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
							Vector3 max = Vector3(FLT_MIN, FLT_MIN, FLT_MIN);

							for (int j = 0; j < 8; ++j)
							{
								Vector3 trf = Vector3::Transform(corners[j], view);
								min = Vector3(std::min(min.x, trf.x), std::min(min.y, trf.y), std::min(min.z, trf.z));
								max = Vector3(std::max(max.x, trf.x), std::max(max.y, trf.y), std::max(max.z, trf.z));
							}

							constexpr float zMult = 10.0f;
							if (min.z < 0)
							{
								min.z *= zMult;
							}
							else
							{
								min.z /= zMult;
							}
							if (max.z < 0)
							{
								max.z /= zMult;
							}
							else
							{
								max.z *= zMult;
							}

							float size = std::max(max.x - min.x, max.y - min.y);
							float depth = max.z - min.z;
							float halfSize = size / 2;
							float halfDepth = depth / 2;
							Matrix projection = Matrix::CreateOrthographicOffCenter(-halfSize, halfSize, -halfSize, halfSize, min.z, max.z);
							
							Matrix viewProjection = view * projection;
							light->m_WorldToShadow[i] = viewProjection;
							light->m_ShadowCascades[i] = Vector4(center.x, center.y, center.z, std::pow(halfSize, 2));

							GetOrthographicPlanes(viewProjection.Invert(), &lightCullerInfo.nearPlane, &lightCullerInfo.farPlane, &lightCullerInfo.rightPlane, &lightCullerInfo.leftPlane, &lightCullerInfo.topPlane, &lightCullerInfo.bottomPlane);

							lightCullerInfo.index = i;
							cullerInfos.emplace_back(lightCullerInfo);
						}
					}
					else
					{
						Matrix view = LightHelper::GetViewMatrix(light);
						Matrix projection = LightHelper::GetProjectionMatrix(light);

						Frustum lightFrustum;
						lightFrustum.CreateFromMatrix(lightFrustum, projection, false);
						lightFrustum.Transform(lightFrustum, transform->GetLocalToWorldMatrix());

						lightFrustum.GetPlanes(&lightCullerInfo.nearPlane, &lightCullerInfo.farPlane, &lightCullerInfo.rightPlane, &lightCullerInfo.leftPlane, &lightCullerInfo.topPlane, &lightCullerInfo.bottomPlane);
						light->m_WorldToShadow[0] = view * projection;

						cullerInfos.emplace_back(lightCullerInfo);
					}
				}
			}
		}

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			CullingResults::RendererInfo info = {};
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(component.second);
			if (meshRenderer->GetMesh() == nullptr)
			{
				continue;
			}
			AABB bounds = meshRenderer->GetBounds();
			info.renderer = meshRenderer;
			info.type = MeshRenderer::Type;
			info.bounds = bounds;

			for (int i = 0; i < cullerInfos.size(); ++i)
			{
				CullingResults::CullerInfo cullerInfo = cullerInfos[i];
				if (bounds.ContainedBy(cullerInfo.nearPlane, cullerInfo.farPlane, cullerInfo.rightPlane, cullerInfo.leftPlane, cullerInfo.topPlane, cullerInfo.bottomPlane))
				{
					info.frustumMask |= 1 << i;
				}
			}

			if (info.frustumMask > 0)
			{
				results.rendererInfos.emplace_back(info);
			}
		}

		results.cullerInfos = cullerInfos;

		s_LastCullerInfo = std::make_pair(nullptr, 0);
	}

	void RenderContext::BindCamera(CullingResults& results)
	{
		Camera* camera = results.camera;
		PerCameraDataConstantBuffer::BindData(camera);
		GatherOperations(results, camera);
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
			VertexLayout layout = {};
			layout.Append(VertexLayout::ElementType::Index);
			GfxDevice::CreateVertexBuffer(layout, INSTANCE_BUFFER_SIZE, s_IndexBuffer);
			uint32_t indices[INSTANCE_BUFFER_SIZE];
			for (uint32_t i = 0; i < INSTANCE_BUFFER_SIZE; ++i)
			{
				indices[i] = i;
			}
			s_IndexBuffer->SetData((float*)indices, INSTANCE_BUFFER_SIZE);
		}

		// Draw meshes
		uint32_t operationCount = s_DrawingOperations.size();
		for (uint32_t i = 0; i < operationCount;)
		{
			auto& operation = s_DrawingOperations[i];
			if (operation.submeshIndex == 255)
			{
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, passIndex, s_IndexBuffer, i, operation.instanceCount));
			}
			else
			{
				SubMeshData* subMesh = operation.mesh->GetSubMesh(operation.submeshIndex);
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, subMesh->GetIndexCount(), subMesh->GetIndexStart(), operation.mesh->GetVertexCount(), passIndex, s_IndexBuffer, i, operation.instanceCount));
			}
			i += operation.instanceCount;
		}
	}
}
