#pragma once
#include "Editor\Serialization\AssetImporter.h"

namespace Blueberry
{
	class TextureImporter : public AssetImporter
	{
		OBJECT_DECLARATION(TextureImporter)

	public:
		TextureImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;
	};
}