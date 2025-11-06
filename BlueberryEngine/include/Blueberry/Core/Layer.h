#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Layer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Layer(const String& name = "Layer");
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnDraw() {}

		const String& GetName() const;
	protected:
		String m_Name;
	};
}