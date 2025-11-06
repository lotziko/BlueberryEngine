#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class GfxComputeShader;

	class BB_API KernelData : public Data
	{
		DATA_DECLARATION(KernelData)

	public:
		KernelData() = default;
		virtual ~KernelData() = default;

		void SetName(const String& name);

	private:
		String m_Name;
	};

	class BB_API ComputeShaderData : public Data
	{
		DATA_DECLARATION(ComputeShaderData)

	public:
		ComputeShaderData() = default;
		virtual ~ComputeShaderData() = default;

		const KernelData& GetKernel(const uint32_t& index) const;
		const size_t GetKernelCount() const;
		void SetKernels(const List<KernelData>& kernels);

	private:
		List<KernelData> m_Kernels;
	};

	class ComputeShader : public Object
	{
		OBJECT_DECLARATION(ComputeShader)

	public:
		ComputeShader() = default;
		virtual ~ComputeShader();

		void Initialize(const List<void*>& shaders);
		void Initialize(const List<void*>& shaders, const ComputeShaderData& data);

		static ComputeShader* Create(const List<void*>& shaders, const ComputeShaderData& shaderData, ComputeShader* existingShader = nullptr);
		
		GfxComputeShader* GetKernel(const uint8_t& index);

	private:
		ComputeShaderData m_Data;

		List<GfxComputeShader*> m_ComputeShaders;
	};
}