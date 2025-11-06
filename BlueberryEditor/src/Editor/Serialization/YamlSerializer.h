#pragma once

#include "Blueberry\Serialization\Serializer.h"

#include <rapidyaml\ryml.h>
#include <concurrent_vector.h>

namespace Blueberry
{
	class YamlSerializer : public Serializer
	{
	public:
		virtual void Serialize(const String& path) override;
		virtual void Deserialize(const String& path) override;

	protected:
		void SerializeNode(ryml::NodeRef& objectNode, Context context);
		void DeserializeNode(ryml::NodeRef& objectNode, Context context);

	protected:
		concurrency::concurrent_vector<std::string> m_Tags;
	};
}