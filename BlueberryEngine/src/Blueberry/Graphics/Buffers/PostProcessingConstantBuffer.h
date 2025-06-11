#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer;

	class PostProcessingConstantBuffer
	{
	public:
		static void BindData(const float& exposure);

	private:
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
	};
}