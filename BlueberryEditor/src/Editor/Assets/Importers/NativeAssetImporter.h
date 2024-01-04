#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class NativeAssetImporter : public AssetImporter
	{
		OBJECT_DECLARATION(NativeAssetImporter)

	public:
		NativeAssetImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;
	};
}