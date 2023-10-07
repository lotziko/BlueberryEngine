#pragma once

#include "Blueberry\Scene\EnityComponent.h"
#include "Blueberry\Graphics\BaseCamera.h"

namespace Blueberry
{
	class Camera : public Component, public BaseCamera
	{
		OBJECT_DECLARATION(Camera)

	public:
		Camera() = default;
		~Camera() = default;

		virtual std::string ToString() const final;
	};
}