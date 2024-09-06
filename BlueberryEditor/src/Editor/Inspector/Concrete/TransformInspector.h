#pragma once

#include "Editor\Inspector\ObjectInspector.h"

#include <map>

namespace Blueberry
{
	class TransformInspector : public ObjectInspector
	{
	public:
		virtual ~TransformInspector() = default;

		virtual void Draw(Object* object) override;
		virtual void DrawScene(Object* object) override;

	private:
		std::unordered_map<std::intptr_t, Vector3> m_TransformEulerCache;
	};
}