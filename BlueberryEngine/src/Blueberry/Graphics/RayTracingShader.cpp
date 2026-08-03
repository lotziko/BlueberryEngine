#include "Blueberry\Graphics\RayTracingShader.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "..\Graphics\GfxRayTracingShader.h"

namespace Blueberry
{
	DATA_DEFINITION(RayTracingShaderData)
	{
		DEFINE_FIELD(RayTracingShaderData, m_PayloadSize, BindingType::Uint, FieldOptions())
		DEFINE_FIELD(RayTracingShaderData, m_AttributesSize, BindingType::Uint, FieldOptions())
		DEFINE_FIELD(RayTracingShaderData, m_RayRecursionDepth, BindingType::Uint, FieldOptions())
	}

	OBJECT_DEFINITION(RayTracingShader, Object)
	{
		DEFINE_BASE_FIELDS(RayTracingShader, Object)
		DEFINE_FIELD(RayTracingShader, m_Data, BindingType::Data, FieldOptions().SetObjectType(&RayTracingShaderData::Type))
	}

	uint32_t RayTracingShaderData::GetPayloadSize() const
	{
		return m_PayloadSize;
	}

	void RayTracingShaderData::SetPayloadSize(uint32_t payloadSize)
	{
		m_PayloadSize = payloadSize;
	}

	uint32_t RayTracingShaderData::GetAttributesSize() const
	{
		return m_AttributesSize;
	}

	void RayTracingShaderData::SetAttributesSize(uint32_t attributesSize)
	{
		m_AttributesSize = attributesSize;
	}

	uint32_t RayTracingShaderData::GetRayRecursionDepth() const
	{
		return m_RayRecursionDepth;
	}

	void RayTracingShaderData::SetRayRecursionDepth(uint32_t rayRecursionDepth)
	{
		m_RayRecursionDepth = rayRecursionDepth;
	}

	const RayTracingShaderData& RayTracingShader::GetData() const
	{
		return m_Data;
	}

	void RayTracingShader::Initialize(const ByteData& shaderBlob)
	{
		GfxDevice::CreateRayTracingShader(shaderBlob, m_RayTracingShader);
	}

	void RayTracingShader::Initialize(const ByteData& shaderBlob, const RayTracingShaderData& data)
	{
		Initialize(shaderBlob);
		m_Data = data;
	}

	GfxRayTracingShader* RayTracingShader::Get() const
	{
		return m_RayTracingShader;
	}

	RayTracingShader* RayTracingShader::Create(const ByteData& shaderBlob, const RayTracingShaderData& data)
	{
		RayTracingShader* shader = Object::Create<RayTracingShader>();
		shader->Initialize(shaderBlob, data);
		return shader;
	}
}