#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class FontImporter : public AssetImporter
	{
		OBJECT_DECLARATION(FontImporter)

	public:
		FontImporter() = default;

	protected:
		virtual void ImportData() final;
	};
}