#include "MeshPreview.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	void MeshPreview::Draw(Mesh* mesh, GfxTexture* target)
	{
		if (m_Scene == nullptr)
		{
			Shader* meshPreviewShader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/MeshPreview.shader"));
			m_MeshPreviewMaterial = Material::Create(meshPreviewShader);

			m_Scene = new Scene();

			Entity* meshEntity = m_Scene->CreateEntity("Mesh");
			m_Renderer = meshEntity->AddComponent<MeshRenderer>();
			m_Renderer->SetMaterial(m_MeshPreviewMaterial);

			Entity* cameraEntity = m_Scene->CreateEntity("Camera");
			m_Camera = cameraEntity->AddComponent<Camera>();
			m_Camera->SetOrthographic(false);
			m_Camera->SetAspectRatio(1.0f);
			m_Camera->SetFieldOfView(60.0f);
			m_Camera->SetPixelSize(Vector2(target->GetWidth(), target->GetHeight()));
		}
	}
}
