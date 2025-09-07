#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class MeshRendererEditor : public ObjectEditor
	{
	public:
		virtual ~MeshRendererEditor() = default;

		virtual void OnDrawScene() override;
	};
}