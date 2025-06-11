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

float3 TransformObjectToWorld(float3 positionOS)
{
	return mul(float4(positionOS, 1.0f), OBJECT_TO_WORLD_MATRIX).xyz;
}

float4 TransformWorldToClip(float3 positionWS)
{
	return mul(float4(positionWS, 1.0f), VIEW_PROJECTION_MATRIX);
}

float4 TransformObjectToClip(float3 positionOS)
{
	return mul(mul(float4(positionOS, 1.0f), OBJECT_TO_WORLD_MATRIX), VIEW_PROJECTION_MATRIX);
}

float3 TransformObjectToWorldNormal(float3 normalOS)
{
	return normalize(mul(float4(normalOS, 0.0f), OBJECT_TO_WORLD_MATRIX).xyz);
}

float Linearize01Depth(float depth, float2 params)
{
	return 1.0 / (params.x * depth + params.y);
}

float3 ReconstructNormal(float3 normal)
{
	return normalize(float3(normal.x, normal.y, sqrt(saturate(1 - dot(normal.xy, normal.xy)))));
}

#endif