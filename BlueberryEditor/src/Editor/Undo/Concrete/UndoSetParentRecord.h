#pragma once

#include "Blueberry\Core\ObjectPtr.h"

#include "..\UndoRecord.h"

namespace Blueberry
{
	class Transform;

	class UndoSetParentRecord : public UndoRecord
	{
	public:
		UndoSetParentRecord(Transform* transform, Transform* newParent, bool worldPositionStays, size_t siblingIndex);

		virtual void OnUndo() final;
		virtual void OnRedo() final;

	private:
		ObjectPtr<Transform> m_Transform;
		ObjectPtr<Transform> m_BeforeParent;
		ObjectPtr<Transform> m_AfterParent;
		size_t m_BeforeSiblingIndex = 0;
		size_t m_AfterSiblingIndex = 0;
	};
}