#ifndef REALTIME_SHADOWS_INCLUDED
#define REALTIME_SHADOWS_INCLUDED

#include "Macros.hlsl"

uint GetFaceIndex(float3 direction)
{
	float faceID;
	if (abs(direction.z) >= abs(direction.x) && abs(direction.z) >= abs(direction.y))
	{
		faceID = (direction.z < 0.0) ? 1 : 0;
	}
	else if (abs(direction.y) >= abs(direction.x))
	{
		faceID = (direction.y < 0.0) ? 5 : 4;
	}
	else
	{
		faceID = (direction.x < 0.0) ? 2 : 3;
	}
	return faceID;
}

bool IsOutOfBounds(float4 position, float4 bounds)
{
	float3 lower = float3(position.xy >= bounds.xy, position.z >= 0.0);
	float3 upper = float3(position.xy <= bounds.zw, position.z <= 1.0);
	return lower.x * lower.y * lower.z * upper.x * upper.y * upper.z == 0;
}

float4 TransformWorldToShadow(float3 positionWS, float4x4 worldToShadow)
{
	float4 positionSS = mul(float4(positionWS, 1.0), worldToShadow);
	positionSS.xyz /= positionSS.w;
	return float4(positionSS.xy, positionSS.zw);
}

float ComputeCascadeIndex(float3 positionWS, float4 cascades[3])
{
	float3 sphereToWS0 = positionWS - cascades[0].xyz;
	float3 sphereToWS1 = positionWS - cascades[1].xyz;
	float3 sphereToWS2 = positionWS - cascades[2].xyz;
	float4 weights = float4(dot(sphereToWS0, sphereToWS0) < cascades[0].w, dot(sphereToWS1, sphereToWS1) < cascades[1].w, dot(sphereToWS2, sphereToWS2) < cascades[2].w, 1);
	weights.yzw = saturate(weights.yzw - weights.xyz);
	return half(4.0) - dot(weights, half4(4, 3, 2, 1));
}

float4 ApplyShadowBias(float4 positionSS)
{
	positionSS.z = saturate(positionSS.z - 0.0001);
	return positionSS;
}

float4 ApplyShadowBias(float4 positionSS, float3 normalWS, float3 lightDirectionWS)
{
	float bias = max(0.0005 * (1.0 - dot(normalWS, lightDirectionWS)), 0.0001);
	positionSS.z = saturate(positionSS.z - bias);
	return positionSS;
}

float ComputeShadowPCF3x3(float4 positionSS, Texture2D shadowmap, SamplerComparisonState shadowmapSampler, float4 shadowTerm0, float4 shadowTerm1, float4 shadowTerm2, float4 shadowTerm3)
{
	float attenuation = 0;
	
	float4 attenuation4;
	attenuation4.x = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm1.xy, positionSS.z).r;
	attenuation4.y = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm1.zy, positionSS.z).r;
	attenuation4.z = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm1.xw, positionSS.z).r;
	attenuation4.w = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm1.zw, positionSS.z).r;

	attenuation = dot(attenuation4, 0.25);
	if (attenuation == 0.0 || attenuation == 1.0)
	{
		return attenuation;
	}
	attenuation *= shadowTerm0.x * 4.0;

	attenuation4.x = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm2.xz, positionSS.z).r;
	attenuation4.y = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm3.xz, positionSS.z).r;
	attenuation4.z = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm3.zy, positionSS.z).r;
	attenuation4.w = SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy + shadowTerm2.zy, positionSS.z).r;
	attenuation += dot(attenuation4, shadowTerm0.y);

	attenuation += SAMPLE_TEXTURE2D_SHADOW(shadowmap, shadowmapSampler, positionSS.xy, positionSS.z).r * shadowTerm0.z;

	return attenuation;
}

#endif