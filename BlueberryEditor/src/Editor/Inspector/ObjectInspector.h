#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Scene\EnityComponent.h"

namespace Blueberry
{
	class ObjectInspector
	{
	public:
		virtual void Draw(Object* object) = 0;
	};
}