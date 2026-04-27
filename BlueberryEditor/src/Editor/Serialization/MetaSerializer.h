#pragma once

#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class MetaSerializer : public Serializer
	{
	public:
		virtual void Serialize(const String& path, SerializationFlags flags) final;
		virtual void Deserialize(const String& path, SerializationFlags flags) final;
	};
}