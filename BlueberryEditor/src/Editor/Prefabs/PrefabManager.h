#pragma once

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

		static void CreatePrefab(const std::string& path, Entity* entity);
		static void UnpackPrefabInstance(Entity* entity);

	private:
		static std::unordered_map<ObjectId, ObjectId> s_EntityToPrefabInstance;
		static std::unordered_set<ObjectId> s_PrefabEntities;

		friend class PrefabInstance;
	};
}