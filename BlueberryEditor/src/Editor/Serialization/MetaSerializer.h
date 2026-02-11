#pragma once

#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class MetaSerializer : public Serializer
	{
	public:
		virtual void Serialize(const String& path, const bool& isText) final;
		virtual void Deserialize(const String& path) final;
	};
}