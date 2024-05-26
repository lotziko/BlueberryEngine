#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class MaterialInspector : public ObjectInspector
	{
	public:
		virtual ~MaterialInspector() = default;

		virtual void Draw(Object* object) override;
	};
}