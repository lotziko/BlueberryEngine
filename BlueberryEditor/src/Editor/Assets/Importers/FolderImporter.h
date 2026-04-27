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
		virtual bool IsRequiringReimport() const final;
		virtual bool IsImportable() const final;
		virtual void ImportData() final;
	};
}