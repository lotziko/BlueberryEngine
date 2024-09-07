#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class SphereColliderInspector : public ObjectInspector
	{
	public:
		SphereColliderInspector() = default;
		virtual ~SphereColliderInspector() = default;

		virtual void DrawScene(Object* object) override;
	};
}