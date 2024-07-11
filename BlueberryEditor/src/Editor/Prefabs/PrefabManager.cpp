#include "bbpch.h"
#include "PrefabManager.h"

#include "Blueberry\Scene\Entity.h"
#include "Editor\Prefabs\PrefabInstance.h"

namespace Blueberry
{
	std::unordered_map<ObjectId, ObjectId> PrefabManager::s_EntityToPrefabInstance = std::unordered_map<ObjectId, ObjectId>();

	bool PrefabManager::IsPrefabInstace(Entity* entity)
	{
		return s_EntityToPrefabInstance.count(entity->GetObjectId()) > 0;
	}

	PrefabInstance* PrefabManager::GetInstance(Entity* entity)
	{
		auto it = s_EntityToPrefabInstance.find(entity->GetObjectId());
		if (it != s_EntityToPrefabInstance.end())
		{
			Object* object = ObjectDB::GetObject(it->second);
			if (object != nullptr)
			{
				return (PrefabInstance*)object;
			}
		}
		return nullptr;
	}

	PrefabInstance* PrefabManager::CreateInstance(Entity* prefabEntity)
	{
		// Check if entity is prefab
		if (!ObjectDB::HasGuid(prefabEntity))
		{
			return nullptr;
		}

		PrefabInstance* instance = PrefabInstance::Create(prefabEntity);
		return instance;
	}
}
