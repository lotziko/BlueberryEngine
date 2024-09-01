#include "bbpch.h"
#include "PrefabManager.h"

#include "Blueberry\Scene\Entity.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	std::unordered_map<ObjectId, ObjectId> PrefabManager::s_EntityToPrefabInstance = std::unordered_map<ObjectId, ObjectId>();
	std::unordered_set<ObjectId> PrefabManager::s_PrefabEntities = std::unordered_set<ObjectId>();

	bool PrefabManager::IsPrefabInstanceRoot(Entity* entity)
	{
		return s_EntityToPrefabInstance.count(entity->GetObjectId()) > 0;
	}

	bool PrefabManager::IsPartOfPrefabInstance(Entity* entity)
	{
		return s_PrefabEntities.count(entity->GetObjectId()) > 0;
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

	void PrefabManager::CreatePrefab(const std::string& path, Entity* entity)
	{
		std::string prefabName(entity->GetName());
		prefabName.append(".prefab");

		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		relativePath.append(prefabName);

		AssetDB::CreateAsset(entity, relativePath.string());
		AssetDB::SaveAssets();
		AssetDB::Refresh();
	}

	void PrefabManager::UnpackPrefabInstance(Entity* entity)
	{
		auto it = s_EntityToPrefabInstance.find(entity->GetObjectId());
		if (it != s_EntityToPrefabInstance.end())
		{
			Object* object = ObjectDB::GetObject(it->second);
			PrefabInstance* instance = (PrefabInstance*)object;
			instance->RemovePrefabEntities(instance->m_Entity.Get());
			instance->m_Entity = nullptr;
			Object::Destroy(instance);
			s_EntityToPrefabInstance.erase(it->first);
		}
	}
}
