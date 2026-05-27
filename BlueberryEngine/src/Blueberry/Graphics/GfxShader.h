#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxShader
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxShader() = default;

		friend struct GfxDrawingOperation;
		friend class Material;
	};
	
	class GfxVertexShader : public GfxShader
	{
	};

	class GfxGeometryShader : public GfxShader
	{
	};

	class GfxFragmentShader : public GfxShader
	{
	};
}