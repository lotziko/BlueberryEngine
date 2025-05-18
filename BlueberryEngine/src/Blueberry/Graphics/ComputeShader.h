#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class GfxComputeShader;

	class ComputeShader : public Object
	{
		OBJECT_DECLARATION(ComputeShader)

	public:
		ComputeShader() = default;

		void Initialize(void* computeData);
		static ComputeShader* Create(void* computeData);

	private:
		GfxComputeShader* m_Shader;
	};
}