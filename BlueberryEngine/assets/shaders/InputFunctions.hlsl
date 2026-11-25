#ifndef INPUT_FUNCTIONS_INCLUDED
#define INPUT_FUNCTIONS_INCLUDED

#include "Input.hlsl"

float3 GetNormalizedViewDirectionWS(float3 positionWS)
{
	return normalize(CAMERA_POSITION_WS - positionWS);
}

float2 GetNormalizedScreenSpaceUV(float4 positionCS)
{
	return positionCS.xy * CAMERA_SIZE_INV_SIZE.zw;
}

float2 GetRenderTargetUV(float4 positionCS)
{
	return positionCS.xy * RENDER_TARGET_SIZE_INV_SIZE.zw;
}

float3 TransformObjectToWorld(float3 positionOS)
{
	return mul(OBJECT_TO_WORLD_MATRIX, float4(positionOS, 1.0f)).xyz;
}

float4 TransformWorldToClip(float3 positionWS)
{
	return mul(VIEW_PROJECTION_MATRIX, float4(positionWS, 1.0f));
}

float3 TransformWorldToView(float3 positionWS)
{
	return mul(VIEW_MATRIX, float4(positionWS, 1.0f)).xyz;
}

float4 TransformObjectToClip(float3 positionOS)
{
	return mul(VIEW_PROJECTION_MATRIX, mul(OBJECT_TO_WORLD_MATRIX, float4(positionOS, 1.0f)));
}

float3 TransformClipToWorld(float3 positionCS)
{
	float4 positionWS = mul(INVERSE_VIEW_PROJECTION_MATRIX, float4(positionCS, 1.0f));
	return positionWS.xyz / positionWS.w;
}

float3 TransformViewToWorld(float3 positionVS)
{
	return mul(INVERSE_VIEW_MATRIX, float4(positionVS, 1.0f)).xyz;
}

float3 TransformClipToView(float3 positionCS)
{
	float4 positionVS = mul(INVERSE_PROJECTION_MATRIX, float4(positionCS, 1.0f));
	return positionVS.xyz / positionVS.w;
}

float3 TransformObjectToWorldNormal(float3 normalOS)
{
	return normalize(mul(normalOS, (float3x3)WORLD_TO_OBJECT_MATRIX));
}

float3 TransformObjectToWorldDirection(float3 directionOS)
{
	return normalize(mul((float3x3)OBJECT_TO_WORLD_MATRIX, directionOS));
}

float3 TransformWorldToViewNormal(float3 normalWS)
{
	return mul(VIEW_MATRIX, float4(normalWS, 0.0f)).xyz;
}

float GetProbeMipLevelFromRoughness(float roughness, float3 NdotR)
{
	float mip1px = REFLECTION_ROUGHEST_MIP - REFLECTION_ROUGHNESS_MIP_SCALE * log2(max(roughness, 0.001));
	return REFLECTION_LOD_COUNT - 1 - mip1px;
}

float GetProbeRoughnessFromMipLevel(float mipLevel)
{
	float mip1px = REFLECTION_LOD_COUNT - 1 - mipLevel;
	return exp2((REFLECTION_ROUGHEST_MIP - mip1px) / REFLECTION_ROUGHNESS_MIP_SCALE);
}

#endif