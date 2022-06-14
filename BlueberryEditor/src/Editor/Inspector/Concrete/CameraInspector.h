#pragma once

#include "Editor\Inspector\ObjectInspector.h"

class CameraInspector : public ObjectInspector
{
public:
	virtual ~CameraInspector() = default;

	virtual void Draw(Object* object) override;
};