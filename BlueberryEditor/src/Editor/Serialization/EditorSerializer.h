#pragma once

#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class Scene;
	class Entity;
	class PrefabInstance;

	class EditorSerializer : public Serializer
	{
	public:
		virtual void Serialize(const String& path, const bool& isText) final;
		virtual void Deserialize(const String& path) final;

		virtual void AddAdditionalObject(const ObjectId& objectId) final;

		void GatherPrefabs(Scene* scene);
		void GatherDependencies(HashSet<Guid>& dependencies);

	private:
		void GatherChildrenPrefabs(Entity* entity);
		void FinalizeObjects();

	private:
		List<PrefabInstance*> m_PrefabInstances;
		bool m_IsPrefabAsset = false;
	};
}