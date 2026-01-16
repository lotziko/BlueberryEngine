#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class PrefabImporter : public AssetImporter
	{
		OBJECT_DECLARATION(PrefabImporter)

	public:
		PrefabImporter() = default;

	protected:
		virtual void ImportData() override;
	};
}