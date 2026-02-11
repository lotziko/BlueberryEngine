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
		static Object* Clone(ObjectMapping& mapping, Object* object);
		static Object* Resolve(ObjectMapping& mapping, List<ObjectId>& removed, Object* object);

	private:
		static Object* CloneObject(HashSet<ObjectId>& visited, ObjectMapping& mapping, Object* object);
	};
}