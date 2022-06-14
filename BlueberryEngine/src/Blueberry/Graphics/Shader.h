#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class Shader : public Object
	{
		OBJECT_DECLARATION(Shader)

	public:
		virtual ~Shader() = default;

		virtual void Bind() const = 0;
		virtual void Unbind() const = 0;
	};
}