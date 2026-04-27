#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ClassDB.h"

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
		static Dictionary<TypeId, ObjectEditorInfo>& GetInfos();
		static const ObjectEditorInfo* GetInfo(TypeId type);

		template<class ObjectEditorType>
		static void Register(size_t typeHash);

	private:
		template<class ObjectEditorType>
		static ObjectEditor* CreateObjectEditor()
		{
			return new ObjectEditorType();
		}

	private:
		static Dictionary<TypeId, ObjectEditorInfo> s_ObjectEditors;
	};

	template<class ObjectEditorType>
	inline void ObjectEditorDB::Register(size_t typeHash)
	{
		ObjectEditorInfo info;
		info.createInstance = &ObjectEditorDB::CreateObjectEditor<ObjectEditorType>;
		s_ObjectEditors.insert_or_assign(ClassDB::GetTypeId(typeHash), info);
	}

	#define REGISTER_OBJECT_EDITOR( inspectorType, objectType ) ObjectEditorDB::Register<inspectorType>(TO_HASH(TO_STRING(objectType)));
}