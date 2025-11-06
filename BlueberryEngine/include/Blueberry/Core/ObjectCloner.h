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

	private:
		static Object* CloneObject(ObjectMapping& mapping, Object* object);
	};
}