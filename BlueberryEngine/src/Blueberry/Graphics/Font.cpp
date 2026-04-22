#include "Blueberry\Graphics\Font.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Font, Object)
	{
		DEFINE_BASE_FIELDS(Font, Object)
		DEFINE_FIELD(Font, m_RawData, BindingType::ByteData, FieldOptions())
	}

	const ByteData& Font::GetData() const
	{
		return m_RawData;
	}

	void Font::SetData(const ByteData& data)
	{
		m_RawData = data;
	}
}