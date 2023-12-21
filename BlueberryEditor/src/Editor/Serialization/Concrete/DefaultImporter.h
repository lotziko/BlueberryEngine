#pragma once
#include "Editor\Serialization\AssetImporter.h"

namespace Blueberry
{
	class DefaultImporter : public AssetImporter
	{
		OBJECT_DECLARATION(DefaultImporter)

	public:
		DefaultImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;
	};
}