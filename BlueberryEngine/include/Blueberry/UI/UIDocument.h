#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class BB_API UIDocument : public Object
	{
		OBJECT_DECLARATION(UIDocument)

	public:
		UIDocument() = default;
		virtual ~UIDocument() = default;

		const String& GetData() const;
		void SetData(const String& data);

		void AddDependency(Object* object);
		void ClearDependencies();

		size_t GetUpdateCount() const;

	private:
		String m_Data = {};
		List<ObjectPtr<Object>> m_Dependencies = {};
		size_t m_UpdateCount = 0;
	};
}