#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class SkyRendererInspector : public ObjectInspector
	{
	public:
		SkyRendererInspector() = default;
		virtual ~SkyRendererInspector() = default;

		virtual void Draw(Object* object) override;
	};
}