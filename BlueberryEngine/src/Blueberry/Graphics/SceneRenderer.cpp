#include "bbpch.h"
#include "SceneRenderer.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	void SceneRenderer::Draw(const Ref<Scene>& scene)
	{
		//Update cameras and render
		{
			for (auto component : scene->GetIterator<Camera>())
			{
				auto camera = static_cast<Camera*>(component.second);
				camera->Update();
				Draw(scene, camera);
			}
		}
	}


	void SceneRenderer::Draw(const Ref<Scene>& scene, Camera* camera)
	{
		Draw(scene, camera->GetViewMatrix(), camera->GetProjectionMatrix());
	}

	void SceneRenderer::Draw(const Ref<Scene>& scene, const Matrix& viewMatrix, const Matrix& projectionMatrix)
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
				g_Renderer2D->Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture().get(), spriteRenderer->GetColor());
			}
			g_Renderer2D->End();
		}
	}
}