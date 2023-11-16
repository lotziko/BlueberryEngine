#pragma once
#include "Editor\Serialization\AssetImporter.h"

namespace Blueberry
{
	class TextureImporter : public AssetImporter
	{
		OBJECT_DECLARATION(TextureImporter)

	public:
		TextureImporter() = default;

	protected:
		virtual void SerializeMeta(YAML::Emitter& out) override;
		virtual void DeserializeMeta(YAML::Node& in) override;
		virtual void ImportData() override;
	};
}