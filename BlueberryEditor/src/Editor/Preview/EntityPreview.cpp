#include "EntityPreview.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Core\ObjectCloner.h"

namespace Blueberry
{
	void EntityPreview::Draw(Entity* entity, GfxTexture* target)
	{
		if (m_Entity.Get() != entity)
		{
			if (m_Scene == nullptr)
			{
				m_Scene = new Scene();

				for (int i = 0; i < 360; i += 120)
				{
					Entity* lightEntity = m_Scene->CreateEntity("Light");
					lightEntity->GetTransform()->SetLocalRotation(Quaternion::CreateFromYawPitchRoll(Math::ToRadians(i), Math::ToRadians(15), 0));
					Light* light = lightEntity->AddComponent<Light>();
					light->SetType(LightType::Directional);
					light->SetCastingShadows(false);
					light->SetIntensity(3);
				}

				Entity* cameraEntity = m_Scene->CreateEntity("Camera");
				m_Camera = cameraEntity->AddComponent<Camera>();
				m_Camera->SetOrthographic(false);
				m_Camera->SetNearClipPlane(0.01f);
				m_Camera->SetAspectRatio(1.0f);
				m_Camera->SetFieldOfView(60.0f);
				m_Camera->SetPixelSize(Vector2(static_cast<float>(target->GetWidth()), static_cast<float>(target->GetHeight())));
				m_Camera->SetCameraType(CameraType::Preview);
				m_Camera->SetBackgroundColor(Color(0.0f, 0.0f, 0.0f, 1.0f));

				Entity* skyEntity = m_Scene->CreateEntity("Sky");
				m_SkyRenderer = skyEntity->AddComponent<SkyRenderer>();
				m_SkyRenderer->SetAmbientColor(Color(0.01f, 0.01f, 0.01f, 1));
			}

			AABB bounds = {};
			bool hasBounds = false;
			m_Entity = entity;

			if (!m_Mapping.empty())
			{
				m_Scene->RemoveEntity(m_RootEntity);
				ReleaseMapping();
			}
			GatherMapping(entity);
			m_RootEntity = static_cast<Entity*>(ObjectCloner::Clone(m_Mapping, entity));
			m_Scene->AddEntity(m_RootEntity);

			for (auto& renderer : m_RootEntity->GetComponentsInChildren<Renderer>())
			{
				if (hasBounds)
				{
					bounds.CreateMerged(bounds, bounds, renderer->GetBounds());
				}
				else
				{
					bounds = renderer->GetBounds();
					hasBounds = true;
				}
			}

			if (hasBounds)
			{
				float halfSize = std::max(Vector3(bounds.Extents).Length(), 0.0001f);
				float distance = halfSize * 2.5f;

				Quaternion rotation = Quaternion::CreateFromYawPitchRoll(Math::ToRadians(-120), Math::ToRadians(20), 0);
				Vector3 position = bounds.Center - Vector3::Transform(Vector3::Forward * distance, rotation);

				m_Camera->GetTransform()->SetRotation(rotation);
				m_Camera->GetTransform()->SetPosition(position);
			}
			Entity::Poll();
		}
		DefaultRenderer::Draw(m_Scene, m_Camera, Rectangle(0, 0, target->GetWidth(), target->GetHeight()), target, nullptr);
	}

	void EntityPreview::GatherMapping(Entity* entity)
	{
		Transform* transform = entity->GetTransform();
		Entity* previewEntity = GetOrCreateEntity();
		m_Mapping.insert_or_assign(entity->GetObjectId(), previewEntity->GetObjectId());
		m_Mapping.insert_or_assign(transform->GetObjectId(), previewEntity->GetTransform()->GetObjectId());
		for (auto& component : entity->GetComponents())
		{
			if (component->IsClassType(MeshRenderer::Type))
			{
				MeshRenderer* previewMeshRenderer = GetOrCreateMeshRenderer();
				m_Mapping.insert_or_assign(component->GetObjectId(), previewMeshRenderer->GetObjectId());
			}
			else if (component->IsClassType(SkinnedMeshRenderer::Type))
			{
				SkinnedMeshRenderer* previewSkinnedMeshRenderer = GetOrCreateSkinnedMeshRenderer();
				m_Mapping.insert_or_assign(component->GetObjectId(), previewSkinnedMeshRenderer->GetObjectId());
			}
		}
		for (auto& child : transform->GetChildren())
		{
			GatherMapping(child->GetEntity());
		}
	}

	void EntityPreview::ReleaseMapping()
	{
		for (auto& pair : m_Mapping)
		{
			Object* object = ObjectDB::GetObject(pair.second);
			if (object->IsClassType(Entity::Type))
			{
				m_Entities.push_back(static_cast<Entity*>(object));
			}
			else if (object->IsClassType(MeshRenderer::Type))
			{
				m_MeshRenderers.push_back(static_cast<MeshRenderer*>(object));
			}
			else if (object->IsClassType(SkinnedMeshRenderer::Type))
			{
				m_SkinnedMeshRenderers.push_back(static_cast<SkinnedMeshRenderer*>(object));
			}
		}
		m_Mapping.clear();
	}

	Entity* EntityPreview::GetOrCreateEntity()
	{
		if (m_Entities.empty())
		{
			Entity* entity = m_Scene->CreateEntity("Entity");
			return entity;
		}
		else
		{
			Entity* entity = m_Entities.back();
			m_Entities.pop_back();
			return entity;
		}
	}

	MeshRenderer* EntityPreview::GetOrCreateMeshRenderer()
	{
		if (m_MeshRenderers.empty())
		{
			MeshRenderer* meshRenderer = Object::Create<MeshRenderer>();
			return meshRenderer;
		}
		else
		{
			MeshRenderer* meshRenderer = m_MeshRenderers.back();
			m_MeshRenderers.pop_back();
			return meshRenderer;
		}
	}

	SkinnedMeshRenderer* EntityPreview::GetOrCreateSkinnedMeshRenderer()
	{
		if (m_SkinnedMeshRenderers.empty())
		{
			SkinnedMeshRenderer* skinnedMeshRenderer = Object::Create<SkinnedMeshRenderer>();
			return skinnedMeshRenderer;
		}
		else
		{
			SkinnedMeshRenderer* skinnedMeshRenderer = m_SkinnedMeshRenderers.back();
			m_SkinnedMeshRenderers.pop_back();
			return skinnedMeshRenderer;
		}
	}
}