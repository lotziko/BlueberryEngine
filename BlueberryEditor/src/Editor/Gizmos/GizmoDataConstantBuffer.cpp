#include "GizmoDataConstantBuffer.h"

#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	GfxBuffer* GizmoDataConstantBuffer::s_ConstantBuffer = nullptr;

	struct GizmoData
	{
		Color color;
		float thickness;
		Vector3 dummy;
	};

	static size_t s_GizmoDataId = TO_HASH("GizmoData");

	void GizmoDataConstantBuffer::BindData(const Color& color, float thickness)
	{
		if (s_ConstantBuffer == nullptr)
		{
			BufferProperties constantBufferProperties = {};
			constantBufferProperties.elementCount = 1;
			constantBufferProperties.elementSize = sizeof(GizmoData) * 1;
			constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

			GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
		}

		GizmoData constants = {};
		constants.color = color;
		constants.thickness = thickness;
		s_ConstantBuffer->SetData(reinterpret_cast<char*>(&constants), sizeof(GizmoData));
		GfxDevice::SetGlobalBuffer(s_GizmoDataId, s_ConstantBuffer);
	}
}