#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class MeshRendererInspector : public ObjectInspector
	{
	public:
		virtual ~MeshRendererInspector() = default;

		virtual void DrawScene(Object* object) override;
	};
}