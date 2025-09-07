#include "IconRenderer.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Inspector\ObjectEditor.h"
#include "Editor\Inspector\ObjectEditorDB.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	List<IconRenderer::IconInfo> IconRenderer::s_IconsCache = {};

	bool IconRenderer::Initialize()
	{
		Shader* iconShader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Icon.shader"));
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
		EditorObjectManager::GetEntityCreated().RemoveCallback<&IconRenderer::ClearCache>(); // TODO component created/destroyed instead
		EditorObjectManager::GetEntityDestroyed().RemoveCallback<&IconRenderer::ClearCache>();
		delete s_IconMaterial;
	}

	bool CompareOperations(const IconRenderer::IconInfo& i1, const IconRenderer::IconInfo& i2)
	{
		return i1.distanceToCamera > i2.distanceToCamera;
	}

	void IconRenderer::Draw(Scene* scene, Camera* camera)
	{
		if (s_IconsCache.size() == 0)
		{
			GenerateCache(scene);
		}
		Vector3 cameraPosition = camera->GetTransform()->GetPosition();
		for (auto& info : s_IconsCache)
		{
			info.distanceToCamera = Vector3::DistanceSquared(cameraPosition, info.transform->GetPosition());
		}
		std::sort(s_IconsCache.begin(), s_IconsCache.end(), CompareOperations);

		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, camera->GetTransform()->GetRotation());

		for (auto& info : s_IconsCache)
		{
			if (info.component->GetEntity()->IsActiveInHierarchy())
			{
				Vector3 position = info.transform->GetPosition();
				Matrix modelMatrix = Matrix::CreateBillboard(position, position + cameraDirection, Vector3(0, -1, 0));

				Texture* icon = info.editor->GetIcon(info.component.Get());
				if (icon != nullptr)
				{
					s_IconMaterial->SetTexture("_BaseMap", static_cast<Texture*>(icon));
					PerDrawDataConstantBuffer::BindData(modelMatrix);
					GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_IconMaterial, 0));
					GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_IconMaterial, 1));
				}
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
			for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
			{
				Component* component = entity->GetComponent(i);
				ObjectEditor* editor = ObjectEditor::GetDefaultEditor(component);
				if (editor != nullptr)
				{
					if (editor->GetIcon(component) != nullptr)
					{
						IconInfo info;
						info.transform = entity->GetTransform();
						info.component = component;
						info.editor = editor;
						s_IconsCache.emplace_back(std::move(info));
					}
				}
			}
		}
	}
}
