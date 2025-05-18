#include "MaterialPreview.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\DefaultRenderer.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	void MaterialPreview::Draw(Material* material, RenderTexture* target)
	{
		if (m_Scene == nullptr)
		{
			m_Scene = new Scene();

			Entity* sphereEntity = m_Scene->CreateEntity("Sphere");
			m_Renderer = sphereEntity->AddComponent<MeshRenderer>();
			m_Renderer->SetMesh(StandardMeshes::GetSphere());

			Entity* lightEntity = m_Scene->CreateEntity("Light");
			lightEntity->GetTransform()->SetPosition(Vector3(4, -3, 3));
			Light* light = lightEntity->AddComponent<Light>();
			light->SetType(LightType::Point);
			light->SetCastingShadows(false);
			light->SetRange(20);
			light->SetIntensity(10);

			Vector3 cameraPosition = Vector3(0.4f, -0.05f, 5.0f);
			Vector3 forward = Vector3::Zero - cameraPosition;
			forward.Normalize();

			Entity* cameraEntity = m_Scene->CreateEntity("Camera");
			cameraEntity->GetTransform()->SetPosition(cameraPosition);
			cameraEntity->GetTransform()->SetRotation(LookRotation(forward, Vector3::Down));
			m_Camera = cameraEntity->AddComponent<Camera>();
			m_Camera->SetOrthographic(false);
			m_Camera->SetAspectRatio(1.0f);
			m_Camera->SetFieldOfView(15.0f);
			m_Camera->SetPixelSize(Vector2(target->GetWidth(), target->GetHeight()));
		}
		m_Renderer->SetMaterial(material);
		DefaultRenderer::Draw(m_Scene, m_Camera, Rectangle(0, 0, target->GetWidth(), target->GetHeight()), Color(0, 0, 0, 1), target, nullptr, true);
	}
}
