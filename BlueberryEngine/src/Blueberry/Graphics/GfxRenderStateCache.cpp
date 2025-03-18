#include "bbpch.h"
#include "GfxRenderStateCache.h"

#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	GfxPassData GfxRenderStateCache::GetPassData(Material* material, const uint8_t& passIndex)
	{
		GfxPassData data = {};
		Shader* shader = material->GetShader();
		const PassData* shaderPass = shader->GetData()->GetPass(passIndex);
		if (shaderPass != nullptr)
		{
			uint32_t vertexFlags = 0;
			uint32_t fragmentFlags = 0;
			const List<std::string>& vertexKeywords = shaderPass->GetVertexKeywords();
			const List<std::string>& fragmentKeywords = shaderPass->GetFragmentKeywords();

			if (material->m_ActiveKeywords.size() > 0)
			{
				for (auto& keyword : material->m_ActiveKeywords)
				{
					for (int i = 0; i < vertexKeywords.size(); ++i)
					{
						if (vertexKeywords[i] == keyword)
						{
							vertexFlags |= 1 << i;
						}
					}
					for (int i = 0; i < fragmentKeywords.size(); ++i)
					{
						if (fragmentKeywords[i] == keyword)
						{
							fragmentFlags |= 1 << i;
							break;
						}
					}
				}
			}
			if (Shader::s_ActiveKeywords.size() > 0)
			{
				for (int i = 0; i < vertexKeywords.size(); ++i)
				{
					if (Shader::s_ActiveKeywords.find(TO_HASH(vertexKeywords[i])) != Shader::s_ActiveKeywords.end())
					{
						vertexFlags |= 1 << i;
					}
				}
				for (int i = 0; i < fragmentKeywords.size(); ++i)
				{
					if (Shader::s_ActiveKeywords.find(TO_HASH(fragmentKeywords[i])) != Shader::s_ActiveKeywords.end())
					{
						fragmentFlags |= 1 << i;
						break;
					}
				}
			}
			const ShaderVariant variant = shader->GetVariant(vertexFlags, fragmentFlags, passIndex);
			data.vertexShader = variant.vertexShader;
			data.geometryShader = variant.geometryShader;
			data.fragmentShader = variant.fragmentShader;
			data.cullMode = shaderPass->GetCullMode();
			data.blendSrcColor = shaderPass->GetBlendSrcColor();
			data.blendSrcAlpha = shaderPass->GetBlendSrcAlpha();
			data.blendDstColor = shaderPass->GetBlendDstColor();
			data.blendDstAlpha = shaderPass->GetBlendDstAlpha();
			data.zTest = shaderPass->GetZTest();
			data.zWrite = shaderPass->GetZWrite();
			data.isValid = true;
		}
		else
		{
			data.isValid = false;
		}
		return data;
	}
}