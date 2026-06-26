#pragma once

#include "Blueberry\Graphics\Shader.h"
#include "Concrete\Windows\ComPtr.h"

#include <dxcapi.h>

namespace Blueberry
{
	class HLSLShaderProcessor
	{
	public:
		HLSLShaderProcessor() = default;
		~HLSLShaderProcessor() = default;

		bool Compile(const String& path);
		void SaveVariants(const String& folderPath);
		bool LoadVariants(const String& folderPath);

		const ShaderData& GetShaderData();
		const VariantsData& GetVariantsData();

	private:
		ShaderData m_ShaderData;
		VariantsData m_VariantsData;
	};
}