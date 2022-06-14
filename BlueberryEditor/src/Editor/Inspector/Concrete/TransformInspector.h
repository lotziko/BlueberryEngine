#pragma once

#include "Editor\Inspector\ObjectInspector.h"

#include <map>

class TransformInspector : public ObjectInspector
{
public:
	virtual ~TransformInspector() = default;

	virtual void Draw(Object* object) override;

private:
	std::map<std::intptr_t, Vector3> m_TransformEulerCache;
};