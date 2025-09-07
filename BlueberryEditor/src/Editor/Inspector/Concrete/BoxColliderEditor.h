#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class BoxColliderEditor : public ObjectEditor
	{
	public:
		BoxColliderEditor() = default;
		virtual ~BoxColliderEditor() = default;

		virtual void OnDrawSceneSelected() override;
	};
}