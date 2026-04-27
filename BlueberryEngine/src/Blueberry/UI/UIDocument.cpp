#include "Blueberry\UI\UIDocument.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(UIDocument, Object)
	{
		DEFINE_BASE_FIELDS(UIDocument, Object)
		DEFINE_FIELD(UIDocument, m_Data, BindingType::String, FieldOptions())
		DEFINE_FIELD(UIDocument, m_Dependencies, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Object::Type))
	}

	const String& UIDocument::GetData() const
	{
		return m_Data;
	}

	void UIDocument::SetData(const String& data)
	{
		m_Data = data;
		m_UpdateCount += 1;
	}

	void UIDocument::AddDependency(Object* object)
	{
		m_Dependencies.push_back(object);
	}

	void UIDocument::ClearDependencies()
	{
		m_Dependencies.clear();
	}
	
	size_t UIDocument::GetUpdateCount() const
	{
		return m_UpdateCount;
	}
}