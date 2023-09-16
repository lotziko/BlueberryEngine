#include "bbpch.h"
#include "CameraInspector.h"

#include "Blueberry\Scene\Components\Camera.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void CameraInspector::Draw(Object* object)
	{
		Camera* camera = static_cast<Camera*>(object);
	}
}