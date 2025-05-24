#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxComputeShader
	{
	public:
		virtual ~GfxComputeShader() = default;

	protected:
		Dictionary<size_t, uint8_t> m_ConstantBufferSlots;
		Dictionary<size_t, uint8_t> m_ComputeBufferSlots;
		Dictionary<size_t, std::pair<uint8_t, uint8_t>> m_StructuredBufferSlots;
		Dictionary<size_t, std::pair<uint8_t, uint8_t>> m_TextureSlots;
	};
}