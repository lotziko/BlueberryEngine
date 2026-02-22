#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class SkinnedMeshRendererEditor : public ObjectEditor
	{
	public:
		virtual ~SkinnedMeshRendererEditor() = default;

		virtual void OnDrawSceneSelected() override;
	};
}