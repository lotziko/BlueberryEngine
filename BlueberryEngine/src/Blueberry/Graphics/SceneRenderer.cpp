#include "bbpch.h"
#include "SceneRenderer.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Scene.h"

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


	void SceneRenderer::Draw(Scene* scene, Camera* camera)
	{
		Draw(scene, camera->GetViewMatrix(), camera->GetProjectionMatrix());
	}

	void SceneRenderer::Draw(Scene* scene, const Matrix& viewMatrix, const Matrix& projectionMatrix)
	{
		//Update transforms
		{
			for (auto component : scene->GetIterator<Transform>())
			{
				auto transform = static_cast<Transform*>(component.second);
				transform->Update();
			}
		}
		
		//Draw sprites
		{
			g_Renderer2D->Begin(viewMatrix, projectionMatrix);
			for (auto component : scene->GetIterator<SpriteRenderer>())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				g_Renderer2D->Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), spriteRenderer->GetMaterial(), spriteRenderer->GetColor());
			}
			g_Renderer2D->End();
		}
	}
}