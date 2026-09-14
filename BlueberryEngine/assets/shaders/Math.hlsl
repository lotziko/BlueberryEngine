 #ifndef MATH_INCLUDED
#define MATH_INCLUDED

#include "Input.hlsl"

float3 NormalTSToNormalWS(float3 normalTS, float3 normalWS, float3 tangentWS, float3 bitangentWS)
{
	float3 normal;
	normal.xyz = normalTS.x * tangentWS.xyz;
	normal.xyz += normalTS.y * bitangentWS.xyz;
	normal.xyz += normalTS.z * normalWS.xyz;
	return normalize(normal);
}

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
	return normalize(mul(OBJECT_TO_WORLD_MATRIX, float4(normalOS, 0.0f)).xyz);
}

float3 TransformWorldToViewNormal(float3 normalWS)
{
	return mul(VIEW_MATRIX, float4(normalWS, 0.0f)).xyz;
}

float Linearize01Depth(float depth, float2 params)
{
	return 1.0 / (params.x * depth + params.y);
}

float3 ReconstructNormal(float3 normal)
{
	return normalize(float3(normal.x, normal.y, sqrt(saturate(1 - dot(normal.xy, normal.xy)))));
}

float2 EncodeNormalOctahedral(float3 normal)
{
	normal /= (abs(normal.x) + abs(normal.y) + abs(normal.z));
	if (normal.z < 0.0)
	{
		normal.xy = (1.0 - abs(normal.yx)) * sign(normal.xy);
	}
	return normal.xy * 0.5 + 0.5;
}

float3 DecodeNormalOctahedral(float2 normal)
{
	normal = normal * 2.0 - 1.0;
	float3 result = float3(normal.x, normal.y, 1.0 - abs(normal.x) - abs(normal.y));
	if (result.z < 0.0)
	{
		result.xy = (1.0 - abs(result.yx)) * sign(result.xy);
	}
	return normalize(result);
}

bool IsInsideAABB(float3 position, float3 min, float3 max)
{
	return (position.x > min.x && position.x < max.x && position.y > min.y && position.y < max.y && position.z > min.z && position.z < max.z);
}

#endif