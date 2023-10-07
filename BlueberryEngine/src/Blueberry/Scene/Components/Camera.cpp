#include "bbpch.h"
#include "Camera.h"

#include "Blueberry\Scene\EnityComponent.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Camera)

	std::string Camera::ToString() const
	{
		return "Camera";
	}
}