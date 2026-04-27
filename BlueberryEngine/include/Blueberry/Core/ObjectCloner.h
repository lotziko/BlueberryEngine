#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Object;
	using ObjectMapping = Dictionary<ObjectId, ObjectId>;

	class ObjectCloner
	{
	public:
		static Object* Clone(Object* object);
		static Object* Resolve(ObjectMapping& mapping, List<ObjectId>& removed, Object* object);

	private:
		static void CopyFields(ObjectMapping& mapping, void* source, void* target, const ClassInfo* info);
	};
}