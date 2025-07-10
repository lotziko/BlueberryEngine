#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class BinarySerializer : public Serializer
	{
	public:
		virtual void Serialize(const String& path) override;
		virtual void Deserialize(const String& path) override;

	protected:
		void SerializeNode(std::stringstream& output, Context context);
		void DeserializeNode(std::ifstream& input, Context context);
	};
}