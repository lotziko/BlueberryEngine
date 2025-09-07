#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Entity;

	class PrefabInstance : public Object
	{
		OBJECT_DECLARATION(PrefabInstance)

		PrefabInstance() = default;
		virtual ~PrefabInstance() = default;

		Entity* GetEntity();
		Entity* GetEntity(const FileId& fileId);

		virtual void OnCreate() final;
		virtual void OnDestroy() final;

		static PrefabInstance* Create(Entity* prefab);

	private:
		void AddPrefabEntities(Entity* prefabEntity, Entity* entity);
		void RemovePrefabEntities(Entity* entity);

	private:
		ObjectPtr<Entity> m_Prefab;
		ObjectPtr<Entity> m_Entity;
		Dictionary<FileId, ObjectId> m_FileIdToObject;

		friend class PrefabManager;
	};
}