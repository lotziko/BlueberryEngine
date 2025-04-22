#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class CameraInspector : public ObjectInspector
	{
	public:
		CameraInspector();
		virtual ~CameraInspector() = default;

		virtual Texture* GetIcon(Object* object) final;
		virtual void Draw(Object* object) override;
		virtual void DrawScene(Object* object) override;
	};
}