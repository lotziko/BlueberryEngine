#include "bbpch.h"
#include "Camera.h"

#include "Blueberry\Scene\EnityComponent.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Camera)

	std::string Camera::ToString() const
	{
		return "Camera";
	}

	void Camera::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Camera)
		BIND_FIELD("m_Name", &Camera::m_Name, BindingType::String)
		END_OBJECT_BINDING()
	}
}