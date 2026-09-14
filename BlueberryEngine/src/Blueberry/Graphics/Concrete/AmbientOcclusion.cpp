#include "AmbientOcclusion.h"

#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	ComputeShader* AmbientOcclusion::s_GTAOShader = nullptr;
	GfxBuffer* AmbientOcclusion::s_GTAOData = nullptr;

	static size_t s_GTAODataId = TO_HASH("GTAOData");
	static size_t s_SrcRawDepthId = TO_HASH("_SrcRawDepth");
	static size_t s_OutWorkingDepthMIP0Id = TO_HASH("_OutWorkingDepthMIP0");
	static size_t s_OutWorkingDepthMIP1Id = TO_HASH("_OutWorkingDepthMIP1");
	static size_t s_OutWorkingDepthMIP2Id = TO_HASH("_OutWorkingDepthMIP2");
	static size_t s_OutWorkingDepthMIP3Id = TO_HASH("_OutWorkingDepthMIP3");
	static size_t s_OutWorkingDepthMIP4Id = TO_HASH("_OutWorkingDepthMIP4");
	static size_t s_SrcWorkingDepthId = TO_HASH("_SrcWorkingDepth");
	static size_t s_SrcNormalmapId = TO_HASH("_SrcNormalmap");
	static size_t s_OutWorkingAOTermId = TO_HASH("_OutWorkingAOTerm");
	static size_t s_OutWorkingEdgesId = TO_HASH("_OutWorkingEdges");
	static size_t s_SrcWorkingAOTermId = TO_HASH("_SrcWorkingAOTerm");
	static size_t s_SrcWorkingEdgesId = TO_HASH("_SrcWorkingEdges");
	static size_t s_OutFinalAOTermId = TO_HASH("_OutFinalAOTerm");

	struct GTAOData
	{
		Vector2Int viewportSize;
		Vector2 viewportPixelSize;                  // .zw == 1.0 / ViewportSize.xy

		Vector2 depthUnpackConsts;
		Vector2 cameraTanHalfFOV;

		Vector2 NDCToViewMul;
		Vector2 NDCToViewAdd;

		Vector2 NDCToViewMul_x_PixelSize;
		float effectRadius;                       // world (viewspace) maximum size of the shadow
		float effectFalloffRange;

		float radiusMultiplier;
		float padding0;
		float finalValuePower;
		float denoiseBlurBeta;

		float sampleDistributionPower;
		float thinOccluderCompensation;
		float depthMIPSamplingOffset;
		int noiseIndex;                         // frameIndex % 64 if using TAA or 0 otherwise
	};
	
	void AmbientOcclusion::Initialize()
	{
		s_GTAOShader = static_cast<ComputeShader*>(AssetLoader::Load("assets/shaders/GTAO.compute"));

		BufferProperties gtaoBufferProperties = {};
		gtaoBufferProperties.elementCount = 1;
		gtaoBufferProperties.elementSize = sizeof(GTAOData) * 1;
		gtaoBufferProperties.usageFlags = BufferUsageFlags::ConstantBuffer;

		GfxDevice::CreateBuffer(gtaoBufferProperties, s_GTAOData);
	}

	void AmbientOcclusion::Shutdown()
	{
		Object::Destroy(s_GTAOShader);
		delete s_GTAOData;
	}

	void AmbientOcclusion::Draw(Camera* camera, GfxTexture* depthStencil, GfxTexture* normals, GfxTexture* output, const Rectangle& viewport)
	{
		const Matrix& projection = camera->GetProjectionMatrix();
		float depthLinearizeMul = -projection.m[3][2];
		float depthLinearizeAdd = projection.m[2][2];
		float tanHalfFOVY = 1.0f / projection.m[1][1];
		float tanHalfFOVX = 1.0F / projection.m[0][0];
		Vector2 delta = Vector2(static_cast<float>(viewport.width) / output->GetWidth(), static_cast<float>(viewport.height) / output->GetHeight());

		GTAOData gtaoConstants = {};
		gtaoConstants.viewportSize = Vector2Int(viewport.width, viewport.height);
		gtaoConstants.viewportPixelSize = Vector2(1.0f / viewport.width * delta.x, 1.0f / viewport.height * delta.y);
		gtaoConstants.depthUnpackConsts = Vector2(depthLinearizeMul, depthLinearizeAdd);
		gtaoConstants.cameraTanHalfFOV = Vector2(tanHalfFOVX, tanHalfFOVY);
		gtaoConstants.NDCToViewMul = Vector2(gtaoConstants.cameraTanHalfFOV.x * 2.0f, gtaoConstants.cameraTanHalfFOV.y * -2.0f);
		gtaoConstants.NDCToViewAdd = Vector2(gtaoConstants.cameraTanHalfFOV.x * -1.0f, gtaoConstants.cameraTanHalfFOV.y * 1.0f);
		gtaoConstants.NDCToViewMul_x_PixelSize = Vector2(gtaoConstants.NDCToViewMul.x * gtaoConstants.viewportPixelSize.x, gtaoConstants.NDCToViewMul.y * gtaoConstants.viewportPixelSize.y);
		gtaoConstants.effectRadius = 0.5f;
		gtaoConstants.effectFalloffRange = 0.615f;
		gtaoConstants.radiusMultiplier = 1.457f;
		gtaoConstants.finalValuePower = 2.2f;
		gtaoConstants.denoiseBlurBeta = 1.2f;
		gtaoConstants.sampleDistributionPower = 2.0f;
		gtaoConstants.thinOccluderCompensation = 0.0f;
		gtaoConstants.depthMIPSamplingOffset = 3.30f;

		s_GTAOData->SetData(reinterpret_cast<char*>(&gtaoConstants), sizeof(GTAOData));
		GfxDevice::SetGlobalBuffer(s_GTAODataId, s_GTAOData);

		uint32_t width = depthStencil->GetWidth();
		uint32_t height = depthStencil->GetHeight();
		GfxTexture* workingDepth = GfxTexturePool::Get(width, height, 0, TextureUsageFlags::UnorderedAccess, 1, 5, TextureFormat::R32_Float);
		GfxTexture* workingAOTerm0 = GfxTexturePool::Get(width, height, 0, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R8_UInt);
		GfxTexture* workingAOTerm1 = GfxTexturePool::Get(width, height, 0, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R8_UInt);
		GfxTexture* workingEdges = GfxTexturePool::Get(width, height, 0, TextureUsageFlags::UnorderedAccess, 1, 1, TextureFormat::R32_Float);
		
		GfxDevice::SetGlobalTexture(s_SrcRawDepthId, depthStencil);
		GfxDevice::SetGlobalTexture(s_OutWorkingDepthMIP0Id, workingDepth, 0);
		GfxDevice::SetGlobalTexture(s_OutWorkingDepthMIP1Id, workingDepth, 1);
		GfxDevice::SetGlobalTexture(s_OutWorkingDepthMIP2Id, workingDepth, 2);
		GfxDevice::SetGlobalTexture(s_OutWorkingDepthMIP3Id, workingDepth, 3);
		GfxDevice::SetGlobalTexture(s_OutWorkingDepthMIP4Id, workingDepth, 4);

		uint32_t threadWidth = (viewport.width + 8 - 1) / 8;
		uint32_t threadHeight = (viewport.height + 8 - 1) / 8;

		GfxDevice::SetRenderTarget(nullptr);
		GfxDevice::Dispatch(s_GTAOShader, 0, (viewport.width + 16 - 1) / 16, (viewport.height + 16 - 1) / 16, 1);
		
		GfxDevice::SetGlobalTexture(s_SrcWorkingDepthId, workingDepth);
		GfxDevice::SetGlobalTexture(s_SrcNormalmapId, normals);
		GfxDevice::SetGlobalTexture(s_OutWorkingAOTermId, workingAOTerm0);
		GfxDevice::SetGlobalTexture(s_OutWorkingEdgesId, workingEdges);
		GfxDevice::Dispatch(s_GTAOShader, 1, threadWidth, threadHeight, 1);

		GfxDevice::SetGlobalTexture(s_SrcWorkingAOTermId, workingAOTerm0);
		GfxDevice::SetGlobalTexture(s_SrcWorkingEdgesId, workingEdges);
		GfxDevice::SetGlobalTexture(s_OutFinalAOTermId, workingAOTerm1);
		GfxDevice::Dispatch(s_GTAOShader, 2, threadWidth, threadHeight, 1);

		GfxDevice::SetGlobalTexture(s_SrcWorkingAOTermId, workingAOTerm1);
		GfxDevice::SetGlobalTexture(s_SrcWorkingEdgesId, workingEdges);
		GfxDevice::SetGlobalTexture(s_OutFinalAOTermId, workingAOTerm0);
		GfxDevice::Dispatch(s_GTAOShader, 2, threadWidth, threadHeight, 1);

		GfxDevice::SetGlobalTexture(s_SrcWorkingAOTermId, workingAOTerm0);
		GfxDevice::SetGlobalTexture(s_OutFinalAOTermId, output);
		GfxDevice::Dispatch(s_GTAOShader, 3, threadWidth, threadHeight, 1);

		GfxTexturePool::Release(workingDepth);
		GfxTexturePool::Release(workingAOTerm0);
		GfxTexturePool::Release(workingAOTerm1);
		GfxTexturePool::Release(workingEdges);
	}
}