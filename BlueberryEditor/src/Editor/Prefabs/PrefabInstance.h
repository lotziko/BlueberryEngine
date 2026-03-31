#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class Entity;
	class Transform;

	class PrefabModificationData : public Data
	{
		DATA_DECLARATION(PrefabModificationData)

	public:
		Object* GetTarget();
		void SetTarget(Object* target);

		const String& GetPath();
		void SetPath(const String& path);

		Variant& GetValue();
		void SetValue(const Variant& value);

	private:
		ObjectPtr<Object> m_Target;
		String m_Path;
		Variant m_Value;
	};

	class PrefabAddedEntityData : public Data
	{
		DATA_DECLARATION(PrefabAddedEntityData)

	public:
		Transform* GetParent();
		void SetParent(Transform* parent);

		Entity* GetEntity();
		void SetEntity(Entity* entity);

		const int32_t& GetIndex();
		void SetIndex(const int32_t& index);

	private:
		ObjectPtr<Transform> m_Parent;
		ObjectPtr<Entity> m_Entity;
		int32_t m_Index;
	};

	class PrefabInstance : public Object
	{
		OBJECT_DECLARATION(PrefabInstance)

		PrefabInstance() = default;
		virtual ~PrefabInstance();

		void Initialize();

		Entity* GetEntity();
		Object* GetCorrespondingObject(const FileId& fileId);

		bool HasSource();
		PrefabInstance* GetSource();

		void UpdateIfNeeded();

	private:
		void PrepareData();
		void AddObjectMapping(Object* prefabObject, Object* instanceObject);

	private:
		ObjectPtr<PrefabInstance> m_SourcePrefab;
		ObjectPtr<Transform> m_Parent;
		List<PrefabModificationData> m_Modifications;
		List<PrefabAddedEntityData> m_AddedEntities;

		ObjectPtr<Entity> m_Entity;
		uint32_t m_UpdateCount = 0;

		Dictionary<FileId, ObjectId> m_FileIdToObject;
		Dictionary<ObjectId, ObjectId> m_PrefabToInstanceMapping;
		Dictionary<ObjectId, ObjectId> m_InstanceToPrefabMapping;

		friend class PrefabManager;
		friend class EditorSerializer;
	};
}