#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class AssemblySerializer : public Serializer
	{
	public:
		void Serialize();
		void Deserialize();
	};
}