#pragma once

namespace Blueberry
{
	class PrefabInstance;
	class Entity;

	class PrefabManager
	{
	public:
		static bool IsPrefabInstance(Entity* entity);
		static PrefabInstance* GetInstance(Entity* entity);
		static PrefabInstance* CreateInstance(Entity* entity);

		static void CreatePrefab(const std::string& path, Entity* entity);
		static void UnpackPrefabInstance(Entity* entity);

	private:
		static std::unordered_map<ObjectId, ObjectId> s_EntityToPrefabInstance;

		friend class PrefabInstance;
	};
}