#pragma once

namespace Blueberry
{
	class Object;
	using ObjectMapping = std::unordered_map<ObjectId, ObjectId>;

	class ObjectCloner
	{
	public:
		static Object* Clone(Object* object);

	private:
		static Object* CloneObject(ObjectMapping& mapping, Object* object);
	};
}