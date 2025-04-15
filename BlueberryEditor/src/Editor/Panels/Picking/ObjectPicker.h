#pragma once

namespace Blueberry
{
	class ObjectPicker
	{
	public:
		static void Open(Object** object, const size_t& type);
		static bool GetResult(Object** object);

	private:
		static bool DrawNone(const bool& selected);
		static bool DrawObject(Object* object, const bool& selected);
	};
}