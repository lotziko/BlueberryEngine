#include "bbpch.h"
#include "IconRenderer.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Inspector\ObjectInspector.h"
#include "Editor\Inspector\ObjectInspectorDB.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Light.h"

namespace Blueberry
{
	std::vector<IconRenderer::IconInfo> IconRenderer::s_IconsCache = {};

	bool IconRenderer::Initialize()
	{
		Shader* iconShader = (Shader*)AssetLoader::Load("assets/Icon.shader");
		if (iconShader == nullptr)
		{
			BB_ERROR("Failed to load icon shader.")
				return false;
		}
		s_IconMaterial = Material::Create(iconShader);
		EditorSceneManager::GetSceneLoaded().AddCallback<&IconRenderer::ClearCache>();
		EditorObjectManager::GetEntityCreated().AddCallback<&IconRenderer::ClearCache>();
		EditorObjectManager::GetEntityDestroyed().AddCallback<&IconRenderer::ClearCache>();
		return true;
	}

	void IconRenderer::Shutdown()
	{
		EditorSceneManager::GetSceneLoaded().RemoveCallback<&IconRenderer::ClearCache>();
		EditorObjectManager::GetEntityCreated().RemoveCallback<&IconRenderer::ClearCache>();
		EditorObjectManager::GetEntityDestroyed().RemoveCallback<&IconRenderer::ClearCache>();
		delete s_IconMaterial;
	}

	void IconRenderer::Draw(Scene* scene, BaseCamera* camera)
	{
		if (s_IconsCache.size() == 0)
		{
			GenerateCache(scene);
		}

		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, camera->GetRotation());

		for (auto& info : s_IconsCache)
		{
			Vector3 position = info.transform->GetPosition();
			Matrix modelMatrix = Matrix::CreateBillboard(position, position - cameraDirection, Vector3(0, -1, 0));

			const char* path = info.inspector->GetIconPath(info.component.Get());
			if (path != nullptr)
			{
				s_IconMaterial->SetTexture("_BaseMap", (Texture*)AssetLoader::Load(path));

				PerDrawConstantBuffer::BindData(modelMatrix);
				GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_IconMaterial));
			}
		}
	}

	void IconRenderer::ClearCache()
	{
		s_IconsCache.clear();
	}

	void IconRenderer::GenerateCache(Scene* scene)
	{
		for (auto& pair : scene->GetEntities())
		{
			Entity* entity = pair.second.Get();
			for (auto& component : entity->GetComponents())
			{
				ObjectInspector* inspector = ObjectInspectorDB::GetInspector(component->GetType());
				if (inspector != nullptr)
				{
					if (inspector->GetIconPath(component) != nullptr)
					{
						IconInfo info;
						info.transform = entity->GetTransform();
						info.component = component;
						info.inspector = inspector;
						s_IconsCache.emplace_back(std::move(info));
					}
				}
			}
		}
	}
}
