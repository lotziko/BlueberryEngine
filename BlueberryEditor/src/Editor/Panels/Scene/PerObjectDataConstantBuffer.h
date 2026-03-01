#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer;

	class PerObjectDataConstantBuffer
	{
	public:
		struct CONSTANTS
		{
			Color objectId;
		};

		static void BindData(const Color& indexColor);

	private:
		static GfxBuffer* s_ConstantBuffer;
	};
}