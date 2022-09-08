#pragma once

#include <string>

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

		const std::string& GetName() const;
	protected:
		std::string m_Name;
	};
}