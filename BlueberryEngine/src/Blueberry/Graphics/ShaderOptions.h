#pragma once

#include <map>
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	using RawShaderOptions = std::map<std::string, std::string>;

	class ShaderOptions
	{
	public:
		ShaderOptions() = default;
		ShaderOptions(const RawShaderOptions& rawOptions);
		
		const CullMode& GetCullMode() const;
		const BlendMode& GetBlendSrc() const;
		const BlendMode& GetBlendDst() const;
		const ZWrite& GetZWrite() const;

	private:
		RawShaderOptions m_RawOptions;
		CullMode m_CullMode = CullMode::Front;
		BlendMode m_SrcBlend = BlendMode::One;
		BlendMode m_DstBlend = BlendMode::Zero;
		ZWrite m_ZWrite = ZWrite::On;
	};
}