#include "GfxInputLayoutCacheDX11.h"

#include "Blueberry\Graphics\VertexLayout.h"

#include "..\DX11\GfxDeviceDX11.h"
#include "..\DX11\GfxShaderDX11.h"

namespace Blueberry
{
	void GfxInputLayoutCacheDX11::Shutdown()
	{
		for (auto& pair : m_InputLayouts)
		{
			pair.second->Release();
		}
		m_InputLayouts.clear();
	}

	ID3D11InputLayout* GfxInputLayoutCacheDX11::GetLayout(GfxVertexShaderDX11* shader, VertexLayout* meshLayout)
	{
		size_t key = static_cast<uint64_t>(shader->m_Crc) | (static_cast<uint64_t>(meshLayout->GetCrc()) << 32);
		auto it = m_InputLayouts.find(key);
		if (it != m_InputLayouts.end())
		{
			return it->second;
		}
		else
		{
			for (uint32_t i = 0; i < VERTEX_ATTRIBUTE_COUNT; ++i)
			{
				uint32_t offset = meshLayout->GetOffset(i);
				uint8_t index = shader->m_LayoutIndices[i];
				if (index != UINT8_MAX)
				{
					shader->m_InputElementDescs[index].AlignedByteOffset = offset;
				}
			}
			ID3D11InputLayout* layout = shader->CreateLayout();
			m_InputLayouts.insert_or_assign(key, layout);
			return layout;
		}
	}
}
