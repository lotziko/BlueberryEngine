#include "MaterialPreview.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\DefaultShaders.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	void MaterialPreview::Draw(Material* material, GfxTexture* target)
	{
		if (m_Scene == nullptr)
		{
			m_Scene = new Scene();

			Entity* sphereEntity = m_Scene->CreateEntity("Sphere");
			m_MeshRenderer = sphereEntity->AddComponent<MeshRenderer>();
			m_MeshRenderer->SetMesh(StandardMeshes::GetSphere());

			Entity* lightEntity = m_Scene->CreateEntity("Light");
			lightEntity->GetTransform()->SetPosition(Vector3(4, 3, 3));
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
			cameraEntity->GetTransform()->SetRotation(Math::LookRotation(forward, Vector3::Down));
			m_Camera = cameraEntity->AddComponent<Camera>();
			m_Camera->SetOrthographic(false);
			m_Camera->SetAspectRatio(1.0f);
			m_Camera->SetFieldOfView(15.0f);
			m_Camera->SetPixelSize(Vector2(static_cast<float>(target->GetWidth()), static_cast<float>(target->GetHeight())));
			m_Camera->SetCameraType(CameraType::Preview);
			m_Camera->SetBackgroundColor(Color(0.0f, 0.0f, 0.0f, 1.0f));

			Entity* skyEntity = m_Scene->CreateEntity("Sky");
			m_SkyRenderer = skyEntity->AddComponent<SkyRenderer>();
			m_SkyRenderer->SetAmbientColor(Color(0.01f, 0.01f, 0.01f, 1));
		}
		if (m_Material.Get() != material)
		{
			m_Material = material;
			if (material->GetShader() == DefaultShaders::GetSkybox())
			{
				m_SkyRenderer->SetMaterial(material);
				m_MeshRenderer->GetEntity()->SetActive(false);
			}
			else
			{
				m_MeshRenderer->SetMaterial(material);
				m_MeshRenderer->GetEntity()->SetActive(true);
			}
		}
		DefaultRenderer::Draw(m_Scene, m_Camera, Rectangle(0, 0, target->GetWidth(), target->GetHeight()), target, nullptr);
	}
}
