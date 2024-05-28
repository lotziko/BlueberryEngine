#pragma once

#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class BinarySerializer : public Serializer
	{
	public:
		virtual void Serialize(const std::string& path) override;
		virtual void Deserialize(const std::string& path) override;

	protected:
		void SerializeNode(std::stringstream& output, Context context);
		void DeserializeNode(std::ifstream& input, Context context);
	};
}