#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class GfxRayTracingShader;

	class BB_API RayTracingShaderData : public Data
	{
		DATA_DECLARATION(RayTracingShaderData)

	public:
		RayTracingShaderData() = default;
		virtual ~RayTracingShaderData() = default;

		uint32_t GetPayloadSize() const;
		void SetPayloadSize(uint32_t payloadSize);

		uint32_t GetAttributesSize() const;
		void SetAttributesSize(uint32_t attributesSize);

		uint32_t GetRayRecursionDepth() const;
		void SetRayRecursionDepth(uint32_t rayRecursionDepth);

	private:
		uint32_t m_PayloadSize = 32;
		uint32_t m_AttributesSize = 8;
		uint32_t m_RayRecursionDepth = 1;
	};

	class BB_API RayTracingShader : public Object
	{
		OBJECT_DECLARATION(RayTracingShader)

	public:
		RayTracingShader() = default;
		virtual ~RayTracingShader() = default;

		const RayTracingShaderData& GetData() const;

		void Initialize(const ByteData& shaderBlob);
		void Initialize(const ByteData& shaderBlob, const RayTracingShaderData& data);

		GfxRayTracingShader* Get() const;

		static RayTracingShader* Create(const ByteData& shaderBlob, const RayTracingShaderData& data);

	private:
		RayTracingShaderData m_Data;

		GfxRayTracingShader* m_RayTracingShader = nullptr;
	};
}