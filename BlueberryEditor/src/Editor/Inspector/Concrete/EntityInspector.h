#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class EntityInspector : public ObjectInspector
	{
	public:
		virtual ~EntityInspector() = default;

		virtual void Draw(Object* object) override;
	};
}