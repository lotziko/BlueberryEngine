#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class PrefabInstance;
	class Entity;

	class PrefabManager
	{
	public:
		static bool IsPrefabInstanceRoot(Entity* entity);
		static bool IsPartOfPrefabInstance(Entity* entity);
		static PrefabInstance* GetInstance(Entity* entity);
		static PrefabInstance* CreateInstance(Entity* entity);

		static void CreatePrefab(const String& path, Entity* entity);
		static void UnpackPrefabInstance(Entity* entity);

	private:
		static Dictionary<ObjectId, ObjectId> s_EntityToPrefabInstance;
		static Dictionary<ObjectId, ObjectId> s_PrefabEntities;

		friend class PrefabInstance;
	};
}