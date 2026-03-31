#pragma once

#include "Blueberry\Core\Layer.h"

namespace Blueberry
{
	class EngineLayer : public Layer
	{
	public:
		EngineLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnUpdate() override;
	};
}