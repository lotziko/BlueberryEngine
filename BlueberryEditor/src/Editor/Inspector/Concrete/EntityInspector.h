#pragma once

#include "Editor\Inspector\ObjectInspector.h"

class EntityInspector : public ObjectInspector
{
public:
	virtual ~EntityInspector() = default;

	virtual void Draw(Object* object) override;
};