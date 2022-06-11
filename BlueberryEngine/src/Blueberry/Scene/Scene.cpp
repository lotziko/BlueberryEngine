#include "bbpch.h"
#include "Scene.h"

#include "Blueberry\Core\ServiceContainer.h"

Scene::Scene(const Ref<ServiceContainer>& serviceContainer) : m_ServiceContainer(serviceContainer)
{
}

bool Scene::Initialize()
{
	return true;
}

void Scene::Draw()
{
	//Update cameras and render
	{
		for (auto component : GetIterator<Camera>())
		{
			auto camera = static_cast<Camera*>(component.second);
			camera->Update();
			DrawCamera(camera);
		}
	}
}

void Scene::DrawCamera(Camera* camera)
{
	//Draw sprites
	{
		auto renderer2D = m_ServiceContainer->Renderer2D;
		renderer2D->Begin(camera->GetViewMatrix(), camera->GetProjectionMatrix());
		for (auto component : GetIterator<SpriteRenderer>())
		{
			auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
			renderer2D->Draw(spriteRenderer->GetEntity()->GetTransform()->GetWorldMatrix(), spriteRenderer->GetTexture().get(), spriteRenderer->GetColor());
		}
		renderer2D->End();
	}
}
