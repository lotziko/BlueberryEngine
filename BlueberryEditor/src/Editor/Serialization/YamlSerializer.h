#pragma once

#include <rapidyaml\ryml.h>
#include <concurrent_vector.h>
#include "Blueberry\Serialization\Serializer.h"

namespace Blueberry
{
	class YamlSerializer : public Serializer
	{
	public:
		virtual void Serialize(const std::string& path) override;
		virtual void Deserialize(const std::string& path) override;

	protected:
		void SerializeNode(ryml::NodeRef& objectNode, Context context);
		void DeserializeNode(ryml::NodeRef& objectNode, Context context);

	protected:
		concurrency::concurrent_vector<std::string> m_Tags;
	};
}