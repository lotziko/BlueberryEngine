#include "bbpch.h"
#include "SceneRenderer.h"

#include "Blueberry\Core\ServiceContainer.h"
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
		//Draw sprites
		{
			auto renderer2D = ServiceContainer::Renderer2D;
			renderer2D->Begin(viewMatrix, projectionMatrix);
			for (auto component : scene->GetIterator<SpriteRenderer>())
			{
				auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
				renderer2D->Draw(spriteRenderer->GetEntity()->GetTransform()->GetWorldMatrix(), spriteRenderer->GetTexture().get(), spriteRenderer->GetColor());
			}
			renderer2D->End();
		}
	}
}