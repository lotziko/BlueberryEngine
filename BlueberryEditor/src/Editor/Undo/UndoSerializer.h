#pragma once

#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class UndoSerializer : public Serializer
	{
	public:
		void Serialize(const List<Object*>& objects);
		void Deserialize();
		bool IsEmpty() const;
	};
}