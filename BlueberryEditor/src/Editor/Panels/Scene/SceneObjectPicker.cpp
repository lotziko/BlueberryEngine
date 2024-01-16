#include "bbpch.h"
#include "SceneObjectPicker.h"

#include "Editor\Selection.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\BaseCamera.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Scene\Scene.h"

namespace Blueberry
{
	SceneObjectPicker::SceneObjectPicker()
	{
		TextureProperties properties;
		properties.width = 1920;
		properties.height = 1080;
		properties.data = nullptr;
		properties.type = TextureType::RenderTarget;
		properties.format = TextureFormat::R8G8B8A8_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneRenderTarget);

		properties.width = 1;
		properties.height = 1;
		properties.data = nullptr;
		properties.type = TextureType::Staging;
		GfxDevice::CreateTexture(properties, m_StagingRenderTarget);

		m_SpriteObjectPickerMaterial = Material::Create((Shader*)AssetLoader::Load("assets/SpriteObjectPicker.shader"));
	}

	void SceneObjectPicker::Pick(Scene* scene, BaseCamera& camera, const int& positionX, const int& positionY, const int& viewportWidth, const int& viewportHeight)
	{
		if (scene == nullptr)
		{
			return;
		}

		Rectangle area = Rectangle(Min(Max(positionX, 0), viewportWidth), Min(Max(positionY, 0), viewportHeight), 1, 1);
		char pixel[4];
		int index = 1;

		GfxDevice::SetRenderTarget(m_SceneRenderTarget);
		GfxDevice::SetViewport(0, 0, viewportWidth, viewportHeight);
		GfxDevice::ClearColor({ 0, 0, 0, 0 });
		Renderer2D::Begin();
		for (auto component : scene->GetIterator<SpriteRenderer>())
		{
			auto spriteRenderer = static_cast<SpriteRenderer*>(component.second);
			Renderer2D::Draw(spriteRenderer->GetEntity()->GetTransform()->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), m_SpriteObjectPickerMaterial, Color((float)index / 255.0f, 0, 0), spriteRenderer->GetSortingOrder());
			++index;
		}
		Renderer2D::End();
		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::Copy(m_SceneRenderTarget, m_StagingRenderTarget, area);
		m_StagingRenderTarget->GetPixel(pixel);

		if (pixel[0] > 0)
		{
			index = 1;
			for (auto component : scene->GetIterator<SpriteRenderer>())
			{
				if (index == pixel[0])
				{
					Selection::SetActiveObject(component.second->GetEntity());
					break;
				}
				++index;
			}
		}
		else
		{
			Selection::SetActiveObject(nullptr);
		}
	}
}
