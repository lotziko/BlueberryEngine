#pragma once

namespace Blueberry
{
	class PrefabInstance;
	class Entity;

	class PrefabManager
	{
	public:
		static bool IsPrefabInstace(Entity* entity);
		static PrefabInstance* GetInstance(Entity* entity);
		static PrefabInstance* CreateInstance(Entity* entity);

	private:
		static std::unordered_map<ObjectId, ObjectId> s_EntityToPrefabInstance;

		friend class PrefabInstance;
	};
}