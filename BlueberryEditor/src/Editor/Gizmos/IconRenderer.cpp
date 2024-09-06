#include "bbpch.h"
#include "IconRenderer.h"

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
	bool IconRenderer::Initialize()
	{
		Shader* iconShader = (Shader*)AssetLoader::Load("assets/Icon.shader");
		if (iconShader == nullptr)
		{
			BB_ERROR("Failed to load icon shader.")
				return false;
		}
		s_IconMaterial = Material::Create(iconShader);
		return true;
	}

	void IconRenderer::Shutdown()
	{
		delete s_IconMaterial;
	}

	void IconRenderer::Draw(Scene* scene, BaseCamera* camera)
	{
		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, camera->GetRotation());

		for (auto& pair : scene->GetEntities())
		{
			Entity* entity = pair.second.Get();
			for (auto& component : entity->GetComponents())
			{
				ObjectInspector* inspector = ObjectInspectorDB::GetInspector(component->GetType());
				if (inspector != nullptr)
				{
					const char* path = inspector->GetIconPath(component);
					if (path != nullptr)
					{
						Transform* transform = entity->GetTransform();
						Vector3 position = transform->GetPosition();
						Matrix modelMatrix = Matrix::CreateBillboard(position, position - cameraDirection, Vector3(0, -1, 0));

						s_IconMaterial->SetTexture("_BaseMap", (Texture*)AssetLoader::Load(path));

						PerDrawConstantBuffer::BindData(modelMatrix);
						GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_IconMaterial));
					}
				}
			}
		}
	}
}
