#include "bbpch.h"
#include "CameraInspector.h"

#include "Blueberry\Scene\Components\Camera.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	OBJECT_INSPECTOR_DECLARATION(CameraInspector, Camera)

	void CameraInspector::Draw(Object* object)
	{
		Camera* camera = static_cast<Camera*>(object);

		if (ImGui::CollapsingHeader("Camera"))
		{

		}
	}
}