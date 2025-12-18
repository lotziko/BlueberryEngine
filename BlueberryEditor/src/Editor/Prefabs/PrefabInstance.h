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

	class PrefabInstance : public Object
	{
		OBJECT_DECLARATION(PrefabInstance)

		PrefabInstance() = default;
		virtual ~PrefabInstance() = default;

		Entity* GetEntity();
		Object* GetCorrespondingObject(const FileId& fileId);

		virtual void OnCreate() final;
		virtual void OnDestroy() final;

		static PrefabInstance* Create(Entity* prefab);

	private:
		void Update();
		void AddObjectMapping(Object* prefabObject, Object* instanceObject);
		void Resolve();

	private:
		ObjectPtr<Entity> m_Prefab;
		ObjectPtr<Entity> m_Entity;
		ObjectPtr<Transform> m_Parent;
		List<PrefabModificationData> m_Modifications;

		Dictionary<FileId, ObjectId> m_FileIdToObject;
		Dictionary<ObjectId, ObjectId> m_PrefabToInstanceMapping;
		Dictionary<ObjectId, ObjectId> m_InstanceToPrefabMapping;

		friend class PrefabManager;
		friend class YamlSceneSerializer;
		friend class EditorSceneManager;
	};
}