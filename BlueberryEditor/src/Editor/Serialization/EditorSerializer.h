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
		virtual void Serialize(const String& path, SerializationFlags flags) final;
		virtual void Deserialize(const String& path, SerializationFlags flags) final;

		virtual void AddAdditionalObject(const ObjectId& objectId) final;

		void GatherPrefabs(Scene* scene);
		void GatherDependencies(HashSet<Guid>& dependencies);
		void FinalizeObjects();

	private:
		void GatherChildrenPrefabs(Entity* entity);
		void Finalize(Object* object, const Guid& guid, const FileId& fileId);

	private:
		List<PrefabInstance*> m_PrefabInstances;
		bool m_IsPrefabAsset = false;
	};
}