#pragma once

namespace Blueberry
{
	struct TextureProperties
	{
		UINT width;
		UINT height;
		void* data;
		bool isRenderTarget;
	};
}