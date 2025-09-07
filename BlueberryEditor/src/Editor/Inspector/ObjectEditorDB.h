#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ObjectEditor;

	struct ObjectEditorInfo
	{
		ObjectEditor*(*createInstance)() = nullptr;
	};

	class ObjectEditorDB
	{
	public:
		static Dictionary<size_t, ObjectEditorInfo>& GetInfos();
		static const ObjectEditorInfo& GetInfo(const size_t& type);

		template<class ObjectEditorType>
		static void Register(const size_t& type);

	private:
		template<class ObjectEditorType>
		static ObjectEditor* CreateObjectEditor()
		{
			return new ObjectEditorType();
		}

	private:
		static Dictionary<size_t, ObjectEditorInfo> s_ObjectEditors;
	};

	template<class ObjectEditorType>
	inline void ObjectEditorDB::Register(const size_t& type)
	{
		ObjectEditorInfo info;
		info.createInstance = &ObjectEditorDB::CreateObjectEditor<ObjectEditorType>;
		s_ObjectEditors.insert_or_assign(type, info);
	}

	#define REGISTER_OBJECT_EDITOR( inspectorType, objectType ) ObjectEditorDB::Register<inspectorType>(TO_OBJECT_TYPE(TO_STRING(objectType)));
}