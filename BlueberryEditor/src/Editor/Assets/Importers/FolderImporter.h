#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class FolderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(FolderImporter)

	public:
		FolderImporter() = default;

	protected:
		virtual const bool IsRequiringReimport() override;
		virtual void ImportData() override;
	};
}