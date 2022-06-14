#pragma once

#include "Blueberry\Core\ServiceContainer.h"

namespace Blueberry
{
	class Layer
	{
	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnDraw() {}

		const std::string& GetName() const { return m_Name; }
	protected:
		std::string m_Name;
	};
}