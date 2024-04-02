#include "bbpch.h"
#include "ShaderOptions.h"

namespace Blueberry
{
	CullMode ParseCullMode(const std::string& name)
	{
		if (name == "Front")
		{
			return CullMode::Front;
		}
		else if (name == "Back")
		{
			return CullMode::Back;
		}
		return CullMode::None;
	}

	BlendMode ParseBlendMode(const std::string& name)
	{
		if (name == "Zero")
		{
			return BlendMode::Zero;
		}
		else if (name == "SrcAlpha")
		{
			return BlendMode::SrcAlpha;
		}
		else if (name == "OneMinusSrcAlpha")
		{
			return BlendMode::OneMinusSrcAlpha;
		}
		return BlendMode::One;
	}

	ZWrite ParseZWrite(const std::string& name)
	{
		if (name == "Off")
		{
			return ZWrite::Off;
		}
		return ZWrite::On;
	}

	ShaderOptions::ShaderOptions(const RawShaderOptions& rawOptions)
	{
		auto& cullModeIt = rawOptions.find("Cull");
		if (cullModeIt != rawOptions.end())
		{
			m_CullMode = ParseCullMode(cullModeIt->second);
		}

		auto& blendSrcIt = rawOptions.find("BlendSrc");
		if (blendSrcIt != rawOptions.end())
		{
			m_SrcBlend = ParseBlendMode(blendSrcIt->second);
		}

		auto& blendDstIt = rawOptions.find("BlendDst");
		if (blendSrcIt != rawOptions.end())
		{
			m_DstBlend = ParseBlendMode(blendDstIt->second);
		}

		auto& zWriteIt = rawOptions.find("ZWrite");
		if (zWriteIt != rawOptions.end())
		{
			m_ZWrite = ParseZWrite(zWriteIt->second);
		}
	}

	const CullMode& ShaderOptions::GetCullMode() const
	{
		return m_CullMode;
	}

	const BlendMode& ShaderOptions::GetBlendSrc() const
	{
		return m_SrcBlend;
	}

	const BlendMode& ShaderOptions::GetBlendDst() const
	{
		return m_DstBlend;
	}

	const ZWrite& ShaderOptions::GetZWrite() const
	{
		return m_ZWrite;
	}
}