#include "RenderContext.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Skinning.h"
#include "LightHelper.h"
#include "Blueberry\Graphics\RendererTree.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Threading\JobSystem.h"

#include "Blueberry\Graphics\TextureCube.h"

namespace Blueberry
{
	GfxBuffer* RenderContext::s_IndexBuffer = nullptr;
	size_t RenderContext::s_LastCullingFrame = 0;

	const uint32_t INSTANCE_BUFFER_SIZE = 8192;
	const uint32_t SKINNING_BUFFER_SIZE = 128;
	
	struct DrawingOperation
	{
		Matrix matrix;
		Mesh* mesh;
		ObjectId meshId;
		uint8_t submeshIndex;
		Material* material;
		ObjectId materialId;
		uint8_t instanceCount;
		float distance;
		uint32_t lightmapChartOffset;
		GfxBuffer* vertexBufferOverride;
		bool isCounterClockwise;
	};

	struct CullerInfo
	{
		Object* cullerObject;
		uint32_t index;
		SortingMode sortingMode;
		ObjectsFilter objectsFilter;
	};

	enum class Keyword
	{
		None,
		Lightmap,
		Probes
	};
	
	static List<DrawingOperation> s_DrawingOperations = {};
	static CullerInfo s_LastCullerInfo = {};
	static Object* s_CurrentCuller = nullptr;
	static uint32_t s_CurrentCullerIndex = 0;
	static Keyword s_CurrentKeyword = Keyword::None;
	static List<std::pair<Matrix, Vector4>> s_PerDrawData = {};
	static uint32_t s_Indices[INSTANCE_BUFFER_SIZE];

	static size_t s_LightmapId = TO_HASH("LIGHTMAP");
	static size_t s_ProbesId = TO_HASH("PROBES");

	bool CompareOperationsDefault(const DrawingOperation& o1, const DrawingOperation& o2)
	{
		if (o1.materialId != o2.materialId)
		{
			return o1.materialId < o2.materialId;
		}
		if (o1.meshId != o2.meshId)
		{
			return o1.meshId < o2.meshId;
		}
		if (o1.submeshIndex != o2.submeshIndex)
		{
			return o1.submeshIndex < o2.submeshIndex;
		}
		return o1.distance < o2.distance;
	}

	bool CompareOperationsFrontToBack(const DrawingOperation& o1, const DrawingOperation& o2)
	{
		return o1.distance < o2.distance;
	}

	void BindOperationsRenderers()
	{
		s_PerDrawData.clear();
		uint32_t operationCount = static_cast<uint32_t>(s_DrawingOperations.size());
		for (uint32_t i = 0; i < operationCount; ++i)
		{
			auto& operation = s_DrawingOperations[i];
			s_PerDrawData.push_back(std::move(std::make_pair(GfxDevice::GetGPUMatrix(operation.matrix), Vector4(static_cast<float>(operation.lightmapChartOffset), 0.0f, 0.0f, 0.0f))));
		}
		PerDrawDataConstantBuffer::BindDataInstanced(s_PerDrawData.data(), operationCount);
	}

	void GatherOperations(const CullingResults& results, Object* cullerObject, const uint32_t& index, const SortingMode& sortingMode, const ObjectsFilter& objectsFilter)
	{
		bool isAll = objectsFilter == ObjectsFilter::All;
		bool isStatic = objectsFilter == ObjectsFilter::Static;

		// TODO store data of frame instead of s_LastCullerInfo
		if (s_LastCullerInfo.cullerObject == cullerObject && s_LastCullerInfo.index == index && s_LastCullerInfo.sortingMode == sortingMode && s_LastCullerInfo.objectsFilter == objectsFilter)
		{
			return;
		}
		else
		{
			s_LastCullerInfo = { cullerObject, index, sortingMode, objectsFilter };
		}

		s_DrawingOperations.clear();

		uint32_t cullerIndex = 0;
		uint32_t cullerCount = static_cast<uint32_t>(results.cullerInfos.size());
		Matrix cullerViewMatrix;
		for (uint32_t i = 0; i < cullerCount; ++i)
		{
			auto& cullerInfo = results.cullerInfos[i];
			if (cullerInfo.object == cullerObject && cullerInfo.index == index)
			{
				cullerIndex = i;
				cullerViewMatrix = cullerInfo.viewMatrix;
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
				MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(renderer);
				Transform* transform = meshRenderer->GetTransform();
				if (!isAll && isStatic != transform->IsStatic())
				{
					continue;
				}
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh == nullptr)
				{
					continue;
				}
				Matrix matrix = renderer->GetLocalToWorldMatrix();
				float distance = Vector3::Transform(meshRenderer->GetBounds().Center, cullerViewMatrix).z;
				uint32_t lightmapChartOffset = meshRenderer->GetLightmapChartOffset();
				uint32_t subMeshCount = mesh->GetSubMeshCount();
				Vector3 scale = transform->GetScale();
				bool isCounterClockwise = (scale.x * scale.y * scale.z) < 0.0f;
				if (subMeshCount > 1)
				{
					for (uint32_t i = 0; i < subMeshCount; ++i)
					{
						Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial(i));
						if (material != nullptr)
						{
							DrawingOperation operation = {};
							operation.matrix = matrix;
							operation.mesh = mesh;
							operation.meshId = mesh->GetObjectId();
							operation.submeshIndex = static_cast<uint8_t>(i);
							operation.material = material;
							operation.materialId = material->GetObjectId();
							operation.instanceCount = 1;
							operation.distance = distance;
							operation.lightmapChartOffset = lightmapChartOffset;
							operation.isCounterClockwise = isCounterClockwise;
							s_DrawingOperations.push_back(std::move(operation));
						}
					}
				}
				else
				{
					Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial());
					if (material != nullptr)
					{
						DrawingOperation operation = {};
						operation.matrix = matrix;
						operation.mesh = mesh;
						operation.meshId = mesh->GetObjectId();
						operation.submeshIndex = 255;
						operation.material = material;
						operation.materialId = material->GetObjectId();
						operation.instanceCount = 1;
						operation.distance = distance;
						operation.lightmapChartOffset = lightmapChartOffset;
						operation.isCounterClockwise = isCounterClockwise;
						s_DrawingOperations.push_back(std::move(operation));
					}
				}
			}
			else if (renderer->GetType() == SkinnedMeshRenderer::Type)
			{
				SkinnedMeshRenderer* skinnedMeshRenderer = static_cast<SkinnedMeshRenderer*>(renderer);
				if (isStatic)
				{
					continue;
				}
				if (!skinnedMeshRenderer->HasRoot())
				{
					continue;
				}
				Mesh* mesh = skinnedMeshRenderer->GetMesh();
				if (mesh == nullptr)
				{
					continue;
				}
				Transform* transform = skinnedMeshRenderer->GetTransform();
				Matrix matrix = renderer->GetLocalToWorldMatrix();
				float distance = Vector3::Transform(skinnedMeshRenderer->GetBounds().Center, cullerViewMatrix).z;
				uint32_t subMeshCount = mesh->GetSubMeshCount();
				GfxBuffer* vertexBuffer = Skinning::GetVertexBuffer(skinnedMeshRenderer);
				Vector3 scale = transform->GetScale();
				bool isCounterClockwise = (scale.x * scale.y * scale.z) < 0.0f;
				if (vertexBuffer == nullptr)
				{
					vertexBuffer = mesh->GetVertexBuffer();
				}
				if (subMeshCount > 1)
				{
					for (uint32_t i = 0; i < subMeshCount; ++i)
					{
						Material* material = GfxDrawingOperation::GetValidMaterial(skinnedMeshRenderer->GetMaterial(i));
						if (material != nullptr)
						{
							DrawingOperation operation = {};
							operation.matrix = matrix;
							operation.mesh = mesh;
							operation.meshId = mesh->GetObjectId();
							operation.submeshIndex = static_cast<uint8_t>(i);
							operation.material = material;
							operation.materialId = material->GetObjectId();
							operation.instanceCount = 1;
							operation.distance = distance;
							operation.vertexBufferOverride = vertexBuffer;
							operation.isCounterClockwise = isCounterClockwise;
							s_DrawingOperations.push_back(std::move(operation));
						}
					}
				}
				else
				{
					Material* material = GfxDrawingOperation::GetValidMaterial(skinnedMeshRenderer->GetMaterial());
					if (material != nullptr)
					{
						DrawingOperation operation = {};
						operation.matrix = matrix;
						operation.mesh = mesh;
						operation.meshId = mesh->GetObjectId();
						operation.submeshIndex = 255;
						operation.material = material;
						operation.materialId = material->GetObjectId();
						operation.instanceCount = 1;
						operation.distance = distance;
						operation.vertexBufferOverride = vertexBuffer;
						operation.isCounterClockwise = isCounterClockwise;
						s_DrawingOperations.push_back(std::move(operation));
					}
				}
			}
		}
		if (sortingMode == SortingMode::Default)
		{
			std::sort(s_DrawingOperations.begin(), s_DrawingOperations.end(), CompareOperationsDefault);
		}
		else
		{
			std::sort(s_DrawingOperations.begin(), s_DrawingOperations.end(), CompareOperationsFrontToBack);
		}

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
		results.reflectionProbes.clear();
		results.cullerInfos.clear();

		Matrix view = cameraData.isMultiview ? cameraData.multiviewViewMatrix[0].Invert() : camera->GetInverseViewMatrix();
		Matrix projection = cameraData.isMultiview ? cameraData.multiviewProjectionMatrix[0] : camera->GetProjectionMatrix();
		Frustum cameraFrustum;
		cameraFrustum.CreateFromMatrix(cameraFrustum, projection);
		cameraFrustum.Transform(cameraFrustum, view);

		CullingResults::CullerInfo cameraFrustumInfo = {};
		cameraFrustumInfo.object = camera;
		cameraFrustumInfo.viewMatrix = camera->GetViewMatrix();
		cameraFrustum.GetPlanes(&cameraFrustumInfo.planes[0], &cameraFrustumInfo.planes[1], &cameraFrustumInfo.planes[2], &cameraFrustumInfo.planes[3], &cameraFrustumInfo.planes[4], &cameraFrustumInfo.planes[5]);
		results.cullerInfos.push_back(cameraFrustumInfo);

		results.skyRenderer = nullptr;
		for (auto component : scene->GetIterator<SkyRenderer>())
		{
			results.skyRenderer = static_cast<SkyRenderer*>(component.second);
			break;
		}

		results.probeVolume = nullptr;
		for (auto component : scene->GetIterator<ProbeVolume>())
		{
			results.probeVolume = static_cast<ProbeVolume*>(component.second);
			break;
		}

		// TODO culling
		for (auto component : scene->GetIterator<ReflectionProbe>())
		{
			results.reflectionProbes.push_back(static_cast<ReflectionProbe*>(component.second));
		}

		if (s_LastCullingFrame < Time::GetFrameCount())
		{
			// TODO move to frame start or find the other away to react on the transform movement
			for (auto component : scene->GetIterator<MeshRenderer>())
			{
				auto meshRenderer = static_cast<MeshRenderer*>(component.second);
				meshRenderer->OnPreCull();
			}
			for (auto component : scene->GetIterator<SkinnedMeshRenderer>())
			{
				auto skinnedMeshRenderer = static_cast<SkinnedMeshRenderer*>(component.second);
				skinnedMeshRenderer->OnPreCull();
			}
			for (auto component : scene->GetIterator<Light>())
			{
				auto light = static_cast<Light*>(component.second);
				light->OnPreCull();
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
				results.lights.push_back(light);

				CullingResults::CullerInfo lightCullerInfo = {};
				lightCullerInfo.object = light;
				
				if (light->IsCastingShadows())
				{
					LightType type = light->GetType();
					if (type == LightType::Directional)
					{
						Quaternion rotation = transform->GetRotation();
						Vector3 forward = Vector3::Transform(Vector3::Forward, rotation);
						Vector3 up = Vector3::Transform(Vector3::Up, rotation);
						const float shadowSize = static_cast<float>(LightHelper::GetShadowSize(LightType::Directional));

						if (light->m_IsCached)
						{
							float planes[] = { 10.0f, 25.0f, 60.0f };
							float grid[] = { 1.0f, 5.0f, 12.0f };

							Vector3 center = camera->GetTransform()->GetPosition();
							float zRange = 100.0f;

							for (int i = 0; i < 3; ++i)
							{
								float radius = planes[i];
								Vector3 currentCascadeCenter = camera->m_ShadowCascades[i];
								Vector3 cascadeCenter = center;
								float cascadeGrid = grid[i];

								cascadeCenter.x = std::roundf(cascadeCenter.x * cascadeGrid) / cascadeGrid;
								cascadeCenter.y = std::roundf(cascadeCenter.y * cascadeGrid) / cascadeGrid;
								cascadeCenter.z = std::roundf(cascadeCenter.z * cascadeGrid) / cascadeGrid;

								Vector3 origin = cascadeCenter - forward * (zRange / 2.0f);
								Matrix projection = Matrix::CreateOrthographicOffCenter(-radius, radius, -radius, radius, 0.001f, zRange);
								Matrix view = Matrix::CreateLookAt(origin, cascadeCenter, up);
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

								if (Vector3::Distance(currentCascadeCenter, cascadeCenter) > cascadeGrid * 0.5f)
								{
									light->m_WorldToShadow[i] = viewProjection;
									light->m_ShadowCascades[i] = Vector4(cascadeCenter.x, cascadeCenter.y, cascadeCenter.z, std::powf(radius, 2));
									light->m_IsDirty[i] = true;
								}

								Math::GetOrthographicPlanes(viewProjection.Invert(), &lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);

								lightCullerInfo.index = i;
								lightCullerInfo.viewMatrix = view;
								results.cullerInfos.push_back(std::move(lightCullerInfo));
							}
						}
						else
						{
							float planes[] = { 0.01f, 5.0f, 15.0f, 43.0f };//{ 0.01f, 5.0f, 20.0f, 100.0f };

							for (int i = 0; i < 3; ++i)
							{
								float nearPlane = planes[i];
								float farPlane = planes[i + 1];

								Frustum cameraSliceFrustum;
								cameraSliceFrustum.CreateFromMatrix(cameraSliceFrustum, Matrix::CreatePerspectiveFieldOfView(Math::ToRadians(camera->GetFieldOfView()), camera->GetAspectRatio(), nearPlane, farPlane));
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
								light->m_ShadowCascades[i] = Vector4(center.x, center.y, center.z, std::powf(radius, 2));

								Math::GetOrthographicPlanes(viewProjection.Invert(), &lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);

								lightCullerInfo.index = i;
								lightCullerInfo.viewMatrix = view;
								results.cullerInfos.push_back(std::move(lightCullerInfo));
							}
						}
					}
					else if (type == LightType::Spot)
					{
						Matrix view = LightHelper::GetViewMatrix(light, transform);
						Matrix projection = LightHelper::GetProjectionMatrix(light);

						Frustum lightFrustum;
						lightFrustum.CreateFromMatrix(lightFrustum, projection);
						lightFrustum.Transform(lightFrustum, transform->GetLocalToWorldMatrix());

						lightFrustum.GetPlanes(&lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);
						light->m_WorldToShadow[0] = view * projection;

						lightCullerInfo.viewMatrix = view;
						results.cullerInfos.push_back(std::move(lightCullerInfo));
					}
					else if (type == LightType::Point)
					{
						Matrix projection = LightHelper::GetProjectionMatrix(light, 2.03f);
						for (uint32_t i = 0; i < 6; ++i)
						{
							Matrix view = LightHelper::GetViewMatrix(light, transform, i);
							Frustum lightFrustum;
							lightFrustum.CreateFromMatrix(lightFrustum, projection);
							lightFrustum.Transform(lightFrustum, view.Invert());

							lightFrustum.GetPlanes(&lightCullerInfo.planes[0], &lightCullerInfo.planes[1], &lightCullerInfo.planes[2], &lightCullerInfo.planes[3], &lightCullerInfo.planes[4], &lightCullerInfo.planes[5]);
							light->m_WorldToShadow[i] = view * projection;

							lightCullerInfo.index = i;
							lightCullerInfo.viewMatrix = view;
							results.cullerInfos.push_back(std::move(lightCullerInfo));
						}
					}
				}
			}
		}

		JobSystem::Dispatch(static_cast<uint32_t>(results.cullerInfos.size()), 1, [&results, scene](JobDispatchArgs args)
		{
			scene->GetRendererTree().Cull(results.cullerInfos[args.jobIndex].planes, results.cullerInfos[args.jobIndex].renderers);
		});
		JobSystem::Wait();

		s_LastCullerInfo = { nullptr, 0, SortingMode::Default, ObjectsFilter::All };
	}

	void RenderContext::BindCamera(CullingResults& results, CameraData& cameraData)
	{
		Camera* camera = results.camera;
		PerCameraDataConstantBuffer::BindData(cameraData);
		s_CurrentCuller = camera;
		s_CurrentCullerIndex = 0;
	}

	void RenderContext::DrawSky(CullingResults& results)
	{
		if (results.skyRenderer != nullptr)
		{
			Material* material = results.skyRenderer->GetMaterial();
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
		s_CurrentCuller = light;
		s_CurrentCullerIndex = shadowDrawingSettings.sliceIndex;

		DrawingSettings drawingSettings = {};
		drawingSettings.passIndex = 2; // Shadow caster
		drawingSettings.sortingMode = SortingMode::FrontToBack;
		drawingSettings.objectsFilter = shadowDrawingSettings.objectsFilter;
		DrawRenderers(results, drawingSettings);
	}

	void RenderContext::DrawRenderers(CullingResults& results, DrawingSettings& drawingSettings)
	{
		uint8_t passIndex = drawingSettings.passIndex;

		if (s_IndexBuffer == nullptr)
		{
			for (uint32_t i = 0; i < INSTANCE_BUFFER_SIZE; ++i)
			{
				s_Indices[i] = i;
			}

			BufferProperties indexBufferProperties = {};
			indexBufferProperties.elementCount = INSTANCE_BUFFER_SIZE;
			indexBufferProperties.elementSize = sizeof(uint32_t);
			indexBufferProperties.data = s_Indices;
			indexBufferProperties.dataSize = INSTANCE_BUFFER_SIZE * sizeof(uint32_t);
			indexBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer;

			GfxDevice::CreateBuffer(indexBufferProperties, s_IndexBuffer);
		}

		GatherOperations(results, s_CurrentCuller, s_CurrentCullerIndex, drawingSettings.sortingMode, drawingSettings.objectsFilter);

		// Draw meshes
		uint32_t operationCount = static_cast<uint32_t>(s_DrawingOperations.size());
		for (uint32_t i = 0; i < operationCount;)
		{
			auto& operation = s_DrawingOperations[i];
			Keyword keyword = operation.lightmapChartOffset > 0 ? Keyword::Lightmap : Keyword::Probes;
			if (keyword != s_CurrentKeyword)
			{
				s_CurrentKeyword = keyword;
				if (keyword == Keyword::Lightmap)
				{
					Shader::SetKeyword(s_LightmapId, true);
					Shader::SetKeyword(s_ProbesId, false);
				}
				else
				{
					Shader::SetKeyword(s_LightmapId, false);
					Shader::SetKeyword(s_ProbesId, true);
				}
			}
			if (operation.submeshIndex == 255)
			{
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.vertexBufferOverride, operation.material, passIndex, s_IndexBuffer, i, operation.instanceCount, operation.isCounterClockwise));
			}
			else
			{
				auto& subMesh = operation.mesh->GetSubMesh(operation.submeshIndex);
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.vertexBufferOverride, operation.material, subMesh.GetIndexCount(), subMesh.GetIndexStart(), operation.mesh->GetVertexCount(), passIndex, s_IndexBuffer, i, operation.instanceCount, operation.isCounterClockwise));
			}
			i += operation.instanceCount;
		}
	}
}
