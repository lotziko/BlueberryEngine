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
		~HLSLComputeShaderProcessor();

		bool Compile(const String& path);
		void SaveKernels(const String& folderPath);
		bool LoadKernels(const String& folderPath);

		const ComputeShaderData& GetComputeShaderData();
		const List<void*>& GetShaders();

	private:
		bool Compile(const String& shaderCode, const char* entryPoint, const char* model, ComPtr<ID3DBlob>& blob);

	private:
		ComputeShaderData m_ComputeShaderData;
		List<void*> m_Shaders;
		List<ComPtr<ID3DBlob>> m_Blobs;
	};
}