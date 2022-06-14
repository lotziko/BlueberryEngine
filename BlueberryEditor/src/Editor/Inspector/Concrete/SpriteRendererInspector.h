#pragma once

#include "Editor\Inspector\ObjectInspector.h"

class SpriteRendererInspector : public ObjectInspector
{
public:
	virtual ~SpriteRendererInspector() = default;

	virtual void Draw(Object* object) override;
};