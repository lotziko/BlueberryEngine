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
	return normalize(_CameraForwardDirectionWS - positionWS);
}

#endif