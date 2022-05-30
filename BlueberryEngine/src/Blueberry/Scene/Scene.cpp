#include "bbpch.h"
#include "Scene.h"

#include "Blueberry\Core\ServiceContainer.h"

Scene::Scene(const Ref<ServiceContainer>& serviceContainer) : m_ServiceContainer(serviceContainer)
{
}

bool Scene::Initialize()
{
	AddComponentManager<TransformManager>();
	AddComponentManager<SpriteRendererManager>();
	AddComponentManager<CameraManager>();
	return true;
}

void Scene::Draw()
{
	//Update cameras and render
	{
		for (auto camera : GetIterator<Camera>())
		{
			camera.second->Update();
			DrawCamera(camera.second);
		}
	}
}

void Scene::DrawCamera(Camera* camera)
{
	//Draw sprites
	{
		auto renderer2D = m_ServiceContainer->Renderer2D;
		renderer2D->Begin(camera->GetViewMatrix(), camera->GetProjectionMatrix());
		for (auto spriteRenderer : GetIterator<SpriteRenderer>())
		{
			renderer2D->Draw(spriteRenderer.second->GetEntity()->GetTransform()->GetWorldMatrix(), spriteRenderer.second->GetTexture().get(), spriteRenderer.second->GetColor());
		}
		renderer2D->End();
	}
}
