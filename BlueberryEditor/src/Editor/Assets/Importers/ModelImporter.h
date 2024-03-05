#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class ModelImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ModelImporter)

	public:
		ModelImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;
	};
}