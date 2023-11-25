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
		virtual void Serialize(ryml::NodeRef& node) override final;
		virtual void Deserialize(ryml::NodeRef& node) override final;
		virtual void ImportData() override;
	};
}