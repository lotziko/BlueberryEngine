#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class SpriteRendererInspector : public ObjectInspector
	{
	public:
		virtual ~SpriteRendererInspector() = default;

		virtual void Draw(Object* object) override;
	};
}