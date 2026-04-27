#pragma once

#include "Editor\Inspector\Concrete\AssetImporterEditor.h"

namespace Blueberry
{
	class PrefabImporterEditor : public AssetImporterEditor
	{
	public:
		virtual ~PrefabImporterEditor() = default;

		virtual bool IsInspectorPadded() override;
	};
}