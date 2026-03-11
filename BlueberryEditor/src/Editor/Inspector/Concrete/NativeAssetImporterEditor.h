#pragma once

#include "Editor\Inspector\Concrete\AssetImporterEditor.h"

namespace Blueberry
{
	class NativeAssetImporterEditor : public AssetImporterEditor
	{
	public:
		virtual ~NativeAssetImporterEditor() = default;

		virtual void OnDrawInspector() override;
	};
}