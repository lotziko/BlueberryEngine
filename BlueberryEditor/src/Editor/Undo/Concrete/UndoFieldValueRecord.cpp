#include "UndoFieldValueRecord.h"

#include "Editor\Misc\ObjectHelper.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Misc\VariantHelper.h"
#include "..\UndoManager.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoFieldValueRecord::UndoFieldValueRecord(Object* object, const String& path, const Variant& before, const Variant& after) : m_Object(object), m_Path(path), m_BeforeValue(before), m_AfterValue(after), m_IsPrefab(PrefabManager::IsPartOfPrefabInstance(object))
	{
		if (m_IsPrefab)
		{
			PrefabInstance* instance = PrefabManager::GetInstance(object);
			if (instance->HasSource())
			{
				m_HadModification = PrefabManager::HasModification(object, path);
				PrefabManager::AddModification(object, path, after);
			}
		}
		SceneEvents::GetObjectValueUpdated().Invoke();
	}

	void UndoFieldValueRecord::OnUndo()
	{
		Object* object = m_Object.Get();
		if (m_IsPrefab)
		{
			if (m_HadModification)
			{
				PrefabManager::AddModification(object, m_Path, m_BeforeValue);
				ObjectHelper::WriteValue(object, m_Path, m_BeforeValue);
			}
			else
			{
				PrefabManager::RemoveModification(object, m_Path);
			}
		}
		else
		{
			ObjectHelper::WriteValue(object, m_Path, m_BeforeValue);
		}
		UndoManager::AddDirtyObject(object->GetObjectId());
		SceneEvents::GetObjectValueUpdated().Invoke();
	}

	void UndoFieldValueRecord::OnRedo()
	{
		Object* object = m_Object.Get();
		if (m_IsPrefab)
		{
			PrefabManager::AddModification(object, m_Path, m_AfterValue);
		}
		ObjectHelper::WriteValue(object, m_Path, m_AfterValue);
		UndoManager::AddDirtyObject(object->GetObjectId());
		SceneEvents::GetObjectValueUpdated().Invoke();
	}

	void UndoFieldValueRecord::SetAfterValue(const Variant& after)
	{
		m_AfterValue = after;
		SceneEvents::GetObjectValueUpdated().Invoke();
	}

	bool UndoFieldValueRecord::Equals(Object* object, const String& path) const
	{
		return m_Object == object && m_Path == path;
	}
}