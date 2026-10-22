#pragma once

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