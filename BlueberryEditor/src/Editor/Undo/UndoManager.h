#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Variant.h"
#include "Blueberry\Events\Event.h"

#include "Editor\Serialization\ObjectSnapshot.h"
#include "UndoRecord.h"

namespace Blueberry
{
	class Transform;
	class Entity;
	class Component;

	using UndoEvent = Event<>;
	using RedoEvent = Event<>;
	using ObjectResolveEvent = Event<>;
	enum class BB_API BindingType;

	class UndoManager
	{
	public:
		static void RecordObject(Object* object);
		
		static void SetTransformParent(Transform* transform, Transform* newParent, bool worldPositionStays = true, size_t siblingIndex = 0);
		static void AddEntity(Entity* entity);
		static void AddComponent(Entity* entity, Component* component);
		static void DestroyObject(Object* object);

		static void Undo();
		static void Redo();
		static void Clear();
		static void ClearPlayMode();
		static void IncrementGroup();
		static void ResolveRecordedObjects();

		static UndoEvent& GetUndoed();
		static RedoEvent& GetRedoed();
		static RedoEvent& GetObjectResolved();

	private:
		static void SetObjectValue(Object* object, const String& path, const Variant& before, const Variant& after);
		static void AddDirtyObject(ObjectId id);

	private:
		struct UndoRecordData
		{
			UndoRecord* ptr;
		};

		struct UndoGroupData
		{
			List<UndoRecordData> records;
			bool isPlayMode;
		};

		static bool s_GroupDirty;
		static UndoGroupData s_CurrentGroup;
		static List<UndoGroupData> s_UndoGroups;
		static List<UndoGroupData> s_RedoGroups;
		static HashSet<ObjectId> s_DirtyObjects;

		static Dictionary<ObjectId, ObjectSnapshot*> s_Snapshots;

		static UndoEvent s_Undoed;
		static RedoEvent s_Redoed;
		static ObjectResolveEvent s_ObjectResolved;

		friend class UndoFieldValueRecord;
		friend class UndoSetParentRecord;
		friend class SerializedObject;
	};
}