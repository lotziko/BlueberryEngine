#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Mesh;
	class GfxTexture;
	class Scene;
	class Entity;
	class Camera;
	class SkyRenderer;
	class MeshRenderer;
	class SkinnedMeshRenderer;
	class Transform;

	class EntityPreview
	{
	public:
		void Draw(Entity* entity, GfxTexture* target);

	private:
		void GatherMapping(Entity* entity);
		void ReleaseMapping();

	private:
		Entity* GetOrCreateEntity();
		MeshRenderer* GetOrCreateMeshRenderer();
		SkinnedMeshRenderer* GetOrCreateSkinnedMeshRenderer();

	private:
		Scene* m_Scene;
		ObjectPtr<Entity> m_Entity;
		Camera* m_Camera;
		SkyRenderer* m_SkyRenderer;
		Entity* m_RootEntity;
		List<Entity*> m_Entities;
		List<MeshRenderer*> m_MeshRenderers;
		List<SkinnedMeshRenderer*> m_SkinnedMeshRenderers;
		Dictionary<ObjectId, ObjectId> m_Mapping;
	};
}