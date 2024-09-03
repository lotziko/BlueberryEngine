#include "bbpch.h"
#include "SceneRenderer.h"

#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\PerCameraLightDataConstantBuffer.h"
#include "Blueberry\Scene\Scene.h"

#include "Blueberry\Graphics\Gizmos.h"

namespace Blueberry
{
	void SceneRenderer::Draw(Scene* scene)
	{
		//Update cameras and render
		{
			for (auto component : scene->GetIterator<Camera>())
			{
				auto camera = static_cast<Camera*>(component.second);
				auto transform = camera->GetEntity()->GetTransform();
				if (transform->IsDirty())
				{
					camera->SetPosition(transform->GetLocalPosition());
					camera->SetRotation(transform->GetLocalRotation());
				}
				Draw(scene, camera);
			}
		}
	}

	void SceneRenderer::Draw(Scene* scene, BaseCamera* camera)
	{
		PerCameraDataConstantBuffer::BindData(camera);

		//Update transforms
		{
			for (auto component : scene->GetIterator<Transform>())
			{
				auto transform = static_cast<Transform*>(component.second);
				transform->Update();
			}
		}

		// Bind lights
		{
			std::vector<LightData> lights;
			for (auto component : scene->GetIterator<Light>())
			{
				auto light = static_cast<Light*>(component.second);
				auto transform = light->GetEntity()->GetTransform();
				lights.emplace_back(LightData { transform, light });
			}
			PerCameraLightDataConstantBuffer::BindData(lights);
		}
		
		//Draw sprites
		{
			Renderer2D::Begin();
			for (auto component : scene->GetIterator<SpriteRenderer>())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), spriteRenderer->GetMaterial(), spriteRenderer->GetColor(), spriteRenderer->GetSortingOrder());
			}
			Renderer2D::End();
		}

		Matrix view = camera->GetInverseViewMatrix();
		Matrix projection = camera->GetProjectionMatrix();
		Frustum frustum;
		frustum.CreateFromMatrix(frustum, projection, true);
		frustum.Transform(frustum, view);

		//Draw meshes
		for (auto component : scene->GetIterator<MeshRenderer>())
		{
			auto meshRenderer = static_cast<MeshRenderer*>(component.second);
			AABB bounds = meshRenderer->GetBounds();

			if (frustum.Contains(bounds))
			{
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					Material* material = meshRenderer->GetMaterial();
					PerDrawConstantBuffer::BindData(meshRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix());
					GfxDevice::Draw(GfxDrawingOperation(mesh, material));
				}
			}
		}

		PerDrawConstantBuffer::BindData(Matrix::Identity);
		Gizmos::SetColor(Color(1, 1, 1, 1));
		Gizmos::Begin();
		for (auto component : scene->GetIterator<Light>())
		{
			auto light = static_cast<Light*>(component.second);
			auto transform = light->GetEntity()->GetTransform();
			PerDrawConstantBuffer::BindData(transform->GetLocalToWorldMatrix());
			Gizmos::DrawCircle(Vector3::Zero, light->GetRange());
		}
		Gizmos::End();
	}
}