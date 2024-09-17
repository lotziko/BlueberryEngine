#include "bbpch.h"
#include "SceneRenderer.h"

#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\PerCameraLightDataConstantBuffer.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	void SceneRenderer::Draw(Scene* scene, Camera* camera)
	{
		PerCameraDataConstantBuffer::BindData(camera);

		// Bind lights
		{
			std::vector<LightData> lights;
			for (auto component : scene->GetIterator<Light>())
			{
				auto light = static_cast<Light*>(component.second);
				auto transform = light->GetTransform();
				lights.emplace_back(LightData { transform, light });
			}
			PerCameraLightDataConstantBuffer::BindData(lights);
		}
		
		// Draw sprites
		{
			Renderer2D::Begin();
			for (auto component : scene->GetIterator<SpriteRenderer>())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				Renderer2D::Draw(spriteRenderer->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), spriteRenderer->GetMaterial(), spriteRenderer->GetColor(), spriteRenderer->GetSortingOrder());
			}
			Renderer2D::End();
		}

		Matrix view = camera->GetInverseViewMatrix();
		Matrix projection = camera->GetProjectionMatrix();
		Frustum frustum;
		frustum.CreateFromMatrix(frustum, projection, true);
		frustum.Transform(frustum, view);

		// Draw meshes
		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			auto meshRenderer = static_cast<MeshRenderer*>(component.second);
			AABB bounds = meshRenderer->GetBounds();
			if (frustum.Contains(bounds))
			{
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					PerDrawConstantBuffer::BindData(meshRenderer->GetTransform()->GetLocalToWorldMatrix());
					UINT subMeshCount = mesh->GetSubMeshCount();
					if (subMeshCount > 1)
					{
						for (int i = 0; i < subMeshCount; ++i)
						{
							Material* material = meshRenderer->GetMaterial(i);
							SubMeshData* subMesh = mesh->GetSubMesh(i);
							GfxDevice::Draw(GfxDrawingOperation(mesh, material, subMesh->GetIndexCount(), subMesh->GetIndexStart(), mesh->GetVertexCount()));
						}
					}
					else
					{
						Material* material = meshRenderer->GetMaterial();
						GfxDevice::Draw(GfxDrawingOperation(mesh, material));
					}
				}
			}
		}
	}
}