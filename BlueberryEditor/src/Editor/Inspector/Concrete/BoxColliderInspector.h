#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class BoxColliderInspector : public ObjectInspector
	{
	public:
		BoxColliderInspector() = default;
		virtual ~BoxColliderInspector() = default;

		virtual void DrawScene(Object* object) override;
	};
}