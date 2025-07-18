#include "MeshPreview.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"

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
		m_Renderer->SetMesh(mesh);
		AABB bounds = m_Renderer->GetBounds();
		float distance = std::max(bounds.Extents.x, std::max(bounds.Extents.y, bounds.Extents.z)) * 2;
		Vector3 targetPosition = bounds.Center;
		Vector3 cameraPosition = targetPosition + Vector3(-distance, distance / 3, -distance);
		m_Camera->GetTransform()->SetRotation(Quaternion::CreateFromYawPitchRoll(ToRadians(45), ToRadians(15), 0));
		m_Camera->GetTransform()->SetPosition(cameraPosition);

		DefaultRenderer::Draw(m_Scene, m_Camera, Rectangle(0, 0, target->GetWidth(), target->GetHeight()), Color(0, 0, 0, 1), target, nullptr, true);
	}
}
