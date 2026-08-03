#pragma once

#include "Blueberry\Graphics\RayTracingShader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class HLSLRayTracingShaderProcessor
	{
	public:
		HLSLRayTracingShaderProcessor() = default;
		~HLSLRayTracingShaderProcessor() = default;

		bool Compile(const String& path);
		void Save(const String& folderPath);
		bool Load(const String& folderPath);

		const RayTracingShaderData& GetRayTracingShaderData();
		const ByteData& GetShader();

	private:
		RayTracingShaderData m_RayTracingShaderData;
		ByteData m_Shader;
	};
}