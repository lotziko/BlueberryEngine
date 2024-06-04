#ifndef PBR_INCLUDED
#define PBR_INCLUDED

#include "RealtimeLights.hlsl"

float3 CalculateDirectDiffuse(float3 normalWS, float3 lightDirectionWS, float3 lightColor, float attenuation, float falloff, float2 diffuseExponent)
{
	float diffuseExponentScalar = (diffuseExponent.x + diffuseExponent.y) * 0.5;
	float NdotL = max(0, dot(normalWS, lightDirectionWS));
	return lightColor * pow(NdotL, diffuseExponent.x) * diffuseExponentScalar * attenuation * falloff;
}

float3 CalculatePBR(float3 positionWS, float3 normalWS)
{
	float3 directDiffuse = 0;

	for (int i = 0; i < int(_LightsCount.x); i++)
	{
		float4 lightPositionWS = _LightPosition[i];
		float3 posToLight = lightPositionWS.xyz - positionWS * lightPositionWS.w;
		float distanceSqr = dot(posToLight, posToLight);

		float3 lightDirectionWS = normalize(posToLight);
		float distanceAttenuation = DistanceAttenuation(distanceSqr, _LightAttenuation[i].xy);
		float falloff = LightFalloff(distanceSqr);
		float3 lightColor = _LightColor[i].rgb;
		directDiffuse += CalculateDirectDiffuse(normalWS, lightDirectionWS, lightColor, distanceAttenuation, falloff, 0.6);
	}

	return directDiffuse;
}

#endif