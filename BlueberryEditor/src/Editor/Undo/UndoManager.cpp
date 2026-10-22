#include "UndoManager.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Component.h"

#include "Concrete\UndoSetParentRecord.h"
#include "Concrete\UndoAddEntityRecord.h"
#include "Concrete\UndoAddComponentRecord.h"
#include "Concrete\UndoDestroyEntityRecord.h"
#include "Concrete\UndoDestroyComponentRecord.h"
#include "Concrete\UndoFieldValueRecord.h"

namespace Blueberry
{
	bool UndoManager::s_GroupDirty = false;
	UndoManager::UndoGroupData UndoManager::s_CurrentGroup = {};
	List<UndoManager::UndoGroupData> UndoManager::s_UndoGroups = {};
	List<UndoManager::UndoGroupData> UndoManager::s_RedoGroups = {};
	HashSet<ObjectId> UndoManager::s_DirtyObjects = {};

	UndoEvent UndoManager::s_Undoed = {};
	RedoEvent UndoManager::s_Redoed = {};
	ObjectResolveEvent UndoManager::s_ObjectResolved = {};

	Dictionary<ObjectId, ObjectSnapshot*> UndoManager::s_Snapshots = {};

	void UndoManager::RecordObject(Object* object)
	{
		ObjectId id = object->GetObjectId();
		if (s_Snapshots.count(id) == 0)
		{
			s_Snapshots.insert_or_assign(id, new ObjectSnapshot(object));
		}
	}

	void UndoManager::SetTransformParent(Transform* transform, Transform* newParent, bool worldPositionStays, size_t siblingIndex)
	{
		UndoRecord* ptr = new UndoSetParentRecord(transform, newParent, worldPositionStays, siblingIndex);
		s_CurrentGroup.records.push_back({ ptr });
		s_GroupDirty = true;
	}

	void UndoManager::AddEntity(Entity* entity)
	{
		UndoRecord* ptr = new UndoAddEntityRecord(entity);
		s_CurrentGroup.records.push_back({ ptr });
		s_GroupDirty = true;
	}

	void UndoManager::AddComponent(Entity* entity, Component* component)
	{
		UndoRecord* ptr = new UndoAddComponentRecord(entity, component);
		s_CurrentGroup.records.push_back({ ptr });
		s_GroupDirty = true;
	}

	void UndoManager::DestroyObject(Object* object)
	{
		if (object->GetType() == Entity::Type)
		{
			UndoRecord* ptr = new UndoDestroyEntityRecord(static_cast<Entity*>(object));
			s_CurrentGroup.records.push_back({ ptr });
			s_GroupDirty = true;
		}
		else if (object->IsClassType(Component::Type))
		{
			UndoRecord* ptr = new UndoDestroyComponentRecord(static_cast<Component*>(object));
			s_CurrentGroup.records.push_back({ ptr });
			s_GroupDirty = true;
		}
	}

	void UndoManager::Undo()
	{
		if (Application::IsRunning() && !s_UndoGroups.empty() && !s_UndoGroups.back().isPlayMode)
		{
			return;
		}
		if (s_GroupDirty)
		{
			s_CurrentGroup.isPlayMode = Application::IsRunning();	// This will break assets modified in play mode
			s_UndoGroups.push_back(std::move(s_CurrentGroup));
			s_CurrentGroup = {};
			s_GroupDirty = false;
		}
		if (!s_UndoGroups.empty())
		{
			UndoManager::UndoGroupData& group = s_UndoGroups.back();
			for (auto it = group.records.rbegin(); it != group.records.rend(); ++it)
			{
				it->ptr->OnUndo();
			}
			s_RedoGroups.push_back(std::move(s_UndoGroups.back()));
			s_UndoGroups.pop_back();
			s_Undoed.Invoke();
		}
		if (!s_DirtyObjects.empty())
		{
			for (ObjectId id : s_DirtyObjects)
			{
				Object* object = ObjectDB::GetObject(id);
				if (object != nullptr)
				{
					object->OnPropertiesChanged();
					object->IncrementChangesCount();
				}
			}
			s_DirtyObjects.clear();
		}
	}

	void UndoManager::Redo()
	{
		if (Application::IsRunning() && !s_RedoGroups.empty() && !s_RedoGroups.back().isPlayMode)
		{
			return;
		}
		if (!s_RedoGroups.empty())
		{
			UndoManager::UndoGroupData& group = s_RedoGroups.back();
			for (auto it = group.records.begin(); it != group.records.end(); ++it)
			{
				it->ptr->OnRedo();
			}
			s_UndoGroups.push_back(std::move(s_RedoGroups.back()));
			s_RedoGroups.pop_back();
			s_Redoed.Invoke();
		}
		if (!s_DirtyObjects.empty())
		{
			for (ObjectId id : s_DirtyObjects)
			{
				Object* object = ObjectDB::GetObject(id);
				if (object != nullptr)
				{
					object->OnPropertiesChanged();
					object->IncrementChangesCount();
				}
			}
			s_DirtyObjects.clear();
		}
	}

	void UndoManager::Clear()
	{
		s_UndoGroups.push_back(std::move(s_CurrentGroup));
		for (auto& group : s_UndoGroups)
		{
			for (auto& record : group.records)
			{
				delete record.ptr;
			}
			group.records.clear();
		}
		s_UndoGroups.clear();
		for (auto& group : s_RedoGroups)
		{
			for (auto& record : group.records)
			{
				delete record.ptr;
			}
			group.records.clear();
		}
		s_RedoGroups.clear();
		s_CurrentGroup = {};
		s_GroupDirty = false;
	}

	void UndoManager::ClearPlayMode()
	{
		for (auto it = s_UndoGroups.begin(); it != s_UndoGroups.end();)
		{
			if (it->isPlayMode)
			{
				for (auto& record : it->records)
				{
					delete record.ptr;
				}
				it->records.clear();
				it = s_UndoGroups.erase(it);
			}
			else
			{
				++it;
			}
		}
		for (auto it = s_RedoGroups.begin(); it != s_RedoGroups.end();)
		{
			if (it->isPlayMode)
			{
				for (auto& record : it->records)
				{
					delete record.ptr;
				}
				it->records.clear();
				it = s_RedoGroups.erase(it);
			}
			else
			{
				++it;
			}
		}
		s_CurrentGroup = {};
		s_GroupDirty = false;
	}

	void UndoManager::IncrementGroup()
	{
		if (s_GroupDirty)
		{
			if (!s_RedoGroups.empty())
			{
				s_RedoGroups.clear();
			}
			s_CurrentGroup.isPlayMode = Application::IsRunning();	// This will break assets modified in play mode
			s_UndoGroups.push_back(std::move(s_CurrentGroup));
			s_CurrentGroup = {};
			s_GroupDirty = false;
		}
	}

	void UndoManager::ResolveRecordedObjects()
	{
		if (!s_Snapshots.empty())
		{
			for (auto& pair : s_Snapshots)
			{
				Object* object = ObjectDB::GetObject(pair.first);
				List<FieldValueChange> differences;
				pair.second->Compare(object, differences);
				for (const FieldValueChange& difference : differences)
				{
					UndoRecord* ptr = new UndoFieldValueRecord(object, difference.path, difference.before, difference.after);
					s_CurrentGroup.records.push_back({ ptr });
				}
				object->IncrementChangesCount();
				delete pair.second;
			}
			s_ObjectResolved.Invoke();
			s_Snapshots.clear();
			s_GroupDirty = true;
		}
	}

	UndoEvent& UndoManager::GetUndoed()
	{
		return s_Undoed;
	}

	RedoEvent& UndoManager::GetRedoed()
	{
		return s_Redoed;
	}

	RedoEvent& UndoManager::GetObjectResolved()
	{
		return s_ObjectResolved;
	}

	void UndoManager::SetObjectValue(Object* object, const String& path, const Variant& before, const Variant& after)
	{
		for (auto& record : s_CurrentGroup.records)
		{
			UndoFieldValueRecord* fieldValue = static_cast<UndoFieldValueRecord*>(record.ptr);
			if (fieldValue->Equals(object, path))
			{
				fieldValue->SetAfterValue(after);
				return;
			}
		}
		UndoRecord* ptr = new UndoFieldValueRecord(object, path, before, after);
		s_CurrentGroup.records.push_back({ ptr });
		s_GroupDirty = true;
	}

	void UndoManager::AddDirtyObject(ObjectId id)
	{
		s_DirtyObjects.insert(id);
	}
}