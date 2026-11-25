#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer;

	class GizmoDataConstantBuffer
	{
	public:
		static void BindData(const Color& color, float thickness);

	private:
		static GfxBuffer* s_ConstantBuffer;
	};
}