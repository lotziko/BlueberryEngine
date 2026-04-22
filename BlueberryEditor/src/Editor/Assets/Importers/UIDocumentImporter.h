#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class UIDocumentImporter : public AssetImporter
	{
		OBJECT_DECLARATION(UIDocumentImporter)

	public:
		UIDocumentImporter() = default;

	protected:
		virtual void ImportData() final;
	};
}