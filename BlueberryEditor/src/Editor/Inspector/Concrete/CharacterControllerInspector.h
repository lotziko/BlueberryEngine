#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class CharacterControllerInspector : public ObjectInspector
	{
	public:
		virtual ~CharacterControllerInspector() = default;

		virtual void DrawScene(Object* object) override;
	};
}