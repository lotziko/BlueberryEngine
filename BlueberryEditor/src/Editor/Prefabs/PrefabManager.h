#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class PrefabInstance;
	class Entity;
	class Transform;

	class PrefabManager
	{
	public:
		static bool IsPrefabInstanceRoot(Entity* entity);
		static bool IsPartOfPrefabInstance(Object* object);
		static PrefabInstance* GetInstance(Object* object);
		static PrefabInstance* CreateInstance(Entity* entity);
		static Object* GetCorrespondingPrefabObject(Object* object);

		static void CreatePrefab(const String& path, Entity* entity);
		static void UnpackPrefabInstance(Entity* entity);
		static void AddModification(Object* object, const String& path, Variant& value);
		static void RemoveModification(Object* object, const String& path);
		static const bool HasModification(Object* object, const String& path);
		static void SetParent(Entity* entity, Transform* parent);
		static void RemoveParent(Entity* entity);
		static bool IsPrefabChild(Entity* entity);

	private:
		static Dictionary<ObjectId, ObjectId> s_RootToPrefabInstance;
		static Dictionary<ObjectId, ObjectId> s_ObjectToPrefabInstance;
		static Dictionary<ObjectId, ObjectId> s_ObjectToPrefabObject;

		friend class PrefabInstance;
	};
}