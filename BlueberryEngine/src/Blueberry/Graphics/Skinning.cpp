#include "Blueberry\Graphics\Skinning.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"

namespace Blueberry
{
	static size_t s_VertexSourceId = TO_HASH("_VertexSource");
	static size_t s_VertexResultId = TO_HASH("_VertexResult");
	static size_t s_SkinningDataId = TO_HASH("SkinningData");
	static size_t s_BoneTransformDataId = TO_HASH("_BoneTransformData");

	struct SkinningData
	{
		uint32_t sourceStride;
		uint32_t resultStride;
		uint32_t weightOffset;
		uint32_t vertexCount;
	};

	void Skinning::Initialize()
	{
		s_SkinningShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/Skinning.compute"));

		BufferProperties constantBufferProperties = {};
		constantBufferProperties.elementCount = 1;
		constantBufferProperties.elementSize = sizeof(SkinningData) * 1;
		constantBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;
		GfxDevice::CreateBuffer(constantBufferProperties, s_ConstantBuffer);
	}

	void Skinning::Shutdown()
	{
		Object::Destroy(s_SkinningShader);

		delete s_ConstantBuffer;
		if (s_BoneTransformBuffer != nullptr)
		{
			delete s_BoneTransformBuffer;
		}
	}

	GfxBuffer* Skinning::GetVertexBuffer(SkinnedMeshRenderer* renderer)
	{
		Mesh* mesh = renderer->GetMesh();
		auto& layout = mesh->GetLayout();
		if (layout.Has(VertexAttribute::Normal) && layout.Has(VertexAttribute::Tangent))
		{
			if (renderer->CalculateSkinning())
			{
				uint32_t vertexCount = mesh->GetVertexCount();
				uint32_t sourceStride = layout.GetSize() / sizeof(float);
				uint32_t resultStride = sourceStride - (layout.GetSize(VertexAttribute::BoneWeight) + layout.GetSize(VertexAttribute::BoneIndex)) / sizeof(float);
				uint32_t threadCount = Math::NextDivisableBy(vertexCount, 64) / 64;

				SkinningData skinningConstants = {};
				skinningConstants.sourceStride = sourceStride;
				skinningConstants.resultStride = resultStride;
				skinningConstants.weightOffset = layout.GetOffset(VertexAttribute::BoneWeight) / sizeof(float);
				skinningConstants.vertexCount = vertexCount;
				s_ConstantBuffer->SetData(&skinningConstants, sizeof(SkinningData));
				GfxDevice::SetGlobalBuffer(s_SkinningDataId, s_ConstantBuffer);

				if (!renderer->IsSkinningBufferValid())
				{
					if (renderer->m_SkinningVertexBuffer != nullptr)
					{
						delete renderer->m_SkinningVertexBuffer;
					}
					renderer->m_SkinningMeshId = mesh->GetObjectId();
					renderer->m_SkinningMeshUpdateCount = mesh->GetUpdateCount();

					BufferProperties vertexBufferProperties = {};
					vertexBufferProperties.format = BufferFormat::R32_Float;
					vertexBufferProperties.elementCount = vertexCount;
					vertexBufferProperties.elementSize = resultStride * sizeof(float);
					vertexBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer | BufferUsageFlags::ByteAdressBuffer | BufferUsageFlags::UnorderedAccess;
					GfxDevice::CreateBuffer(vertexBufferProperties, renderer->m_SkinningVertexBuffer);

					GfxDevice::SetGlobalBuffer(s_VertexSourceId, mesh->GetVertexBuffer());
					GfxDevice::SetGlobalBuffer(s_VertexResultId, renderer->m_SkinningVertexBuffer);
					GfxDevice::Dispatch(s_SkinningShader->GetKernel(0), threadCount, 1, 1);
				}

				auto& matrices = renderer->GetSkinningMatrices();
				if (s_BoneTransformBuffer == nullptr || matrices.size() > static_cast<size_t>(s_BoneTransformBuffer->GetElementCount()))
				{
					BufferProperties boneTransformBufferProperties = {};
					boneTransformBufferProperties.elementCount = static_cast<uint32_t>(matrices.size());
					boneTransformBufferProperties.elementSize = sizeof(Matrix) * 1;
					boneTransformBufferProperties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource | BufferUsageFlags::CPUWritable;
					GfxDevice::CreateBuffer(boneTransformBufferProperties, s_BoneTransformBuffer);
				}
				s_BoneTransformBuffer->SetData(matrices.data(), sizeof(Matrix) * matrices.size());

				GfxDevice::SetGlobalBuffer(s_BoneTransformDataId, s_BoneTransformBuffer);
				GfxDevice::SetGlobalBuffer(s_VertexSourceId, mesh->GetVertexBuffer());
				GfxDevice::SetGlobalBuffer(s_VertexResultId, renderer->m_SkinningVertexBuffer);
				GfxDevice::Dispatch(s_SkinningShader->GetKernel(1), threadCount, 1, 1);
			}
			return renderer->m_SkinningVertexBuffer;
		}
		return nullptr;
	}
}
