#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class DefaultImporter : public AssetImporter
	{
		OBJECT_DECLARATION(DefaultImporter)

	public:
		DefaultImporter() = default;

	protected:
		virtual bool IsRequiringReimport() const final;
		virtual bool IsImportable() const final;
		virtual void ImportData() final;
	};
}