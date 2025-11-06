#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class SphereColliderEditor : public ObjectEditor
	{
	public:
		SphereColliderEditor() = default;
		virtual ~SphereColliderEditor() = default;

		virtual void OnDrawSceneSelected() override;
	};
}