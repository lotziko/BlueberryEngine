#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxShader
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxShader() = default;

	protected:
		Dictionary<size_t, uint8_t> m_ConstantBufferSlots;
		Dictionary<size_t, std::pair<uint8_t, uint8_t>> m_StructuredBufferSlots;
		Dictionary<size_t, std::pair<uint8_t, uint8_t>> m_TextureSlots;

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