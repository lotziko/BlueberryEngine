#include "bbpch.h"
#include "RenderContext.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\PerCameraLightDataConstantBuffer.h"

namespace Blueberry
{
	const UINT INSTANCE_BUFFER_SIZE = 2048;

	bool CompareOperations(CullingResults::DrawingOperation o1, CullingResults::DrawingOperation o2)
	{
		return o1.material->GetObjectId() > o2.material->GetObjectId();
	}

	void RenderContext::Cull(Scene* scene, Camera* camera, CullingResults& results)
	{
		if (scene == nullptr || camera == nullptr)
		{
			return;
		}

		results.camera = camera;
		results.lights.clear();
		results.meshRenderers.clear();

		Matrix view = camera->GetInverseViewMatrix();
		Matrix projection = camera->GetProjectionMatrix();
		Frustum frustum;
		frustum.CreateFromMatrix(frustum, projection, true);
		frustum.Transform(frustum, view);

		for (auto component : scene->GetIterator<Light>())
		{
			auto light = static_cast<Light*>(component.second);
			auto transform = light->GetTransform();
			// TODO light types
			Sphere bounds = Sphere(transform->GetPosition(), light->GetRange());
			if (frustum.Contains(bounds))
			{
				results.lights.emplace_back(light);
			}
		}

		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			auto meshRenderer = static_cast<MeshRenderer*>(component.second);
			AABB bounds = meshRenderer->GetBounds();
			if (frustum.Contains(bounds))
			{
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					Matrix matrix = meshRenderer->GetTransform()->GetLocalToWorldMatrix();
					UINT subMeshCount = mesh->GetSubMeshCount();
					if (subMeshCount > 1)
					{
						for (UINT i = 0; i < subMeshCount; ++i)
						{
							Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial(i));
							if (material != nullptr)
							{
								SubMeshData* subMesh = mesh->GetSubMesh(i);
								results.drawingOperations.emplace_back(CullingResults::DrawingOperation{ matrix, mesh, i, material });//results.drawingOperations.emplace_back(std::make_tuple(matrix, material, GfxDrawingOperation(mesh, material, subMesh->GetIndexCount(), subMesh->GetIndexStart(), mesh->GetVertexCount(), 255)));
							}
						}
					}
					else
					{
						Material* material = GfxDrawingOperation::GetValidMaterial(meshRenderer->GetMaterial());
						if (material != nullptr)
						{
							results.drawingOperations.emplace_back(CullingResults::DrawingOperation{ matrix, mesh, 255, material });//results.drawingOperations.emplace_back(std::make_tuple(matrix, material, GfxDrawingOperation(mesh, material, 255)));
						}
					}
				}
			}
		}

		std::sort(results.drawingOperations.begin(), results.drawingOperations.end(), CompareOperations);
	}

	void RenderContext::Bind(CullingResults& results)
	{
		Camera* camera = results.camera;
		PerCameraDataConstantBuffer::BindData(camera);

		// Bind lights
		{
			std::vector<LightData> lights;
			for (Light* light : results.lights)
			{
				auto transform = light->GetTransform();
				lights.emplace_back(LightData{ transform, light });
			}
			PerCameraLightDataConstantBuffer::BindData(lights);
		}

		if (s_IndexBuffer == nullptr)
		{
			VertexLayout layout = {};
			layout.Append(VertexLayout::ElementType::Index);
			GfxDevice::CreateVertexBuffer(layout, INSTANCE_BUFFER_SIZE, s_IndexBuffer);
			uint32_t indices[INSTANCE_BUFFER_SIZE];
			for (UINT i = 0; i < INSTANCE_BUFFER_SIZE; ++i)
			{
				indices[i] = i;
			}
			s_IndexBuffer->SetData((float*)indices, INSTANCE_BUFFER_SIZE);
		}

		// Bind meshes
		{
			UINT operationCount = results.drawingOperations.size();
			Matrix matrices[INSTANCE_BUFFER_SIZE];
			for (UINT i = 0; i < operationCount; ++i)
			{
				matrices[i] = GfxDevice::GetGPUMatrix(results.drawingOperations[i].matrix);
			}
			PerDrawConstantBuffer::BindDataInstanced(matrices, operationCount);
		}
	}

	void RenderContext::Draw(CullingResults& results, DrawingSettings& drawingSettings)
	{
		uint8_t passIndex = drawingSettings.passIndex;

		// Draw meshes


		UINT operationCount = results.drawingOperations.size();
		for (UINT i = 0; i < operationCount; ++i)
		{
			auto& operation = results.drawingOperations[i];
			if (operation.submeshIndex == 255)
			{
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, passIndex, s_IndexBuffer, i));
			}
			else
			{
				SubMeshData* subMesh = operation.mesh->GetSubMesh(operation.submeshIndex);
				GfxDevice::Draw(GfxDrawingOperation(operation.mesh, operation.material, subMesh->GetIndexCount(), subMesh->GetIndexStart(), operation.mesh->GetVertexCount(), passIndex, s_IndexBuffer, i));
			}
		}
	}
}
