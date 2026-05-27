#pragma once

#include "Blueberry\Graphics\ComputeShader.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class HLSLComputeShaderProcessor
	{
	public:
		HLSLComputeShaderProcessor() = default;
		~HLSLComputeShaderProcessor() = default;

		bool Compile(const String& path);
		void SaveKernels(const String& folderPath);
		bool LoadKernels(const String& folderPath);

		const ComputeShaderData& GetComputeShaderData();
		const List<ByteData>& GetShaders();

	private:
		bool Compile(const String& shaderCode, const char* entryPoint, const char* model, ComPtr<ID3DBlob>& blob);

	private:
		ComputeShaderData m_ComputeShaderData;
		List<ByteData> m_Shaders;
	};
}