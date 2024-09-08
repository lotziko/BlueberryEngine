#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class CameraInspector : public ObjectInspector
	{
	public:
		virtual ~CameraInspector() = default;

		virtual const char* GetIconPath(Object* object) override;
		virtual void Draw(Object* object) override;
		virtual void DrawScene(Object* object) override;
	};
}