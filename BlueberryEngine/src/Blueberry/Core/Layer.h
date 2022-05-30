#pragma once

#include "Blueberry\Core\ServiceContainer.h"

class Layer
{
public:
	Layer(const std::string& name = "Layer");
	virtual ~Layer() = default;

	virtual void OnAttach() {}
	virtual void OnDetach() {}
	virtual void OnUpdate() {}
	virtual void OnDraw() {}

	void SetServiceContainer(const Ref<ServiceContainer>& serviceContainer) { m_ServiceContainer = serviceContainer; }
	const std::string& GetName() const { return m_Name; }
protected:
	Ref<ServiceContainer> m_ServiceContainer;
	std::string m_Name;
};

#pragma once
