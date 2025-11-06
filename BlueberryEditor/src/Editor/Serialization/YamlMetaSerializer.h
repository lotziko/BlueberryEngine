#pragma once

#include "Blueberry\Core\Guid.h"
#include "Editor\Serialization\YamlSerializer.h"

namespace Blueberry
{
	class YamlMetaSerializer : public YamlSerializer
	{
	public:
		const Guid& GetGuid();
		void SetGuid(const Guid& guid);

		virtual void Serialize(const String& path) override;
		virtual void Deserialize(const String& path) override;
	private:
		Guid m_Guid;
	};
}