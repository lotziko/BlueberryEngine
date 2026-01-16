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
		virtual const bool IsRequiringReimport() override;
		virtual void ImportData() override;
	};
}