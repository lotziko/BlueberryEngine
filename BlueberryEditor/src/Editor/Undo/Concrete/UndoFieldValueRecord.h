#pragma once

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Variant.h"

#include "..\UndoRecord.h"

namespace Blueberry
{
	enum class BB_API BindingType;

	class UndoFieldValueRecord : public UndoRecord
	{
	public:
		UndoFieldValueRecord(Object* object, const String& path, const Variant& before, const Variant& after);

		virtual void OnUndo() final;
		virtual void OnRedo() final;

		void SetAfterValue(const Variant& after);

		bool Equals(Object* object, const String& path) const;

	private:
		ObjectPtr<Object> m_Object;
		String m_Path;
		Variant m_BeforeValue;
		Variant m_AfterValue;
		bool m_IsPrefab = false;
		bool m_HadModification = false;
	};
}