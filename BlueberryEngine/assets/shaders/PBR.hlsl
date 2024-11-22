#ifndef PBR_INCLUDED
#define PBR_INCLUDED

#include "Structs.hlsl"
#include "RealtimeLights.hlsl"
#include "RealtimeShadows.hlsl"

float CalculateGeometricRoughnessFactor(half3 geometricNormalWs)
{
	half3 normalWsDdx = ddx(geometricNormalWs.xyz);
	half3 normalWsDdy = ddy(geometricNormalWs.xyz);
	return pow(saturate(max(dot(normalWsDdx.xyz, normalWsDdx.xyz), dot(normalWsDdy.xyz, normalWsDdy.xyz))), 0.333);
}

float AdjustRoughnessByGeometricNormal(float roughness, float3 geometricNormalWs)
{
	float geometricRoughnessFactor = CalculateGeometricRoughnessFactor(geometricNormalWs.xyz);
	roughness = max(roughness, geometricRoughnessFactor.x);
	return roughness;
}

half2 CalculateFresnelResponse(half NdotV, half roughness)
{
	return pow(float2(0, 0), 2);
}

void RoughnessEllipseToScaleAndExp(float roughness, out float2 diffuseExponent, out float2 specularExponent, out float2 specularScale)
{
	diffuseExponent = ((1.0 - roughness.xx) * 0.8) + 0.6; // 0.8 and 0.6 are magic numbers
	specularExponent.xy = exp2(pow(float2(1.0, 1.0) - roughness.xx, float2(1.5, 1.5)) * float2(14.0, 14.0)); // Outputs 1-16384
	specularScale.xy = 1.0 - saturate(roughness.xx * 0.5); // This is an energy conserving scalar for the roughness exponent.
}

float3 CalculateDirectDiffuse(float3 normalWS, float3 lightDirectionWS, float3 lightColor, float attenuation, float falloff, float2 diffuseExponent)
{
	float diffuseExponentScalar = (diffuseExponent.x + diffuseExponent.y) * 0.5;
	float NdotL = max(0, dot(normalWS, lightDirectionWS));
	return lightColor * pow(NdotL, diffuseExponent.x) * diffuseExponentScalar * attenuation * falloff;
}

float CalculateDistributionGGX(float3 NDotH, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NDotH2 = NDotH * NDotH;

	float num = a2;
	float denom = (NDotH2 * (a2 - 1.0) + 1.0);
	denom = 1 * denom * denom;

	return num / denom;
}

half CalculateGeometrySchlickGGX(float NDotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float num = NDotV;
	float denom = NDotV * (1.0 - k) + k;

	return num / denom;
}

half3 CalculateFresnelSchlick(float LDotH, float3 reflectance)
{
	return reflectance + (1.0 - reflectance) * pow(1.0 - LDotH, 5);
}

half3 CalculateDirectSpecular(float3 normalWS, float3 viewDirectionWS, float3 lightDirectionWS, float3 lightColor, float attenuation, float falloff, float3 reflectance, float roughness)
{
	float3 halfAngleDirWS = normalize(lightDirectionWS.xyz + viewDirectionWS.xyz);
	float LDotH = max(0, dot(lightDirectionWS.xyz, halfAngleDirWS.xyz));
	float NDotH = max(0, dot(normalWS.xyz, halfAngleDirWS.xyz));
	float NDotL = max(0, dot(normalWS.xyz, lightDirectionWS.xyz));
	float NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	float NDF = CalculateDistributionGGX(NDotH, roughness);
	float G = CalculateGeometrySchlickGGX(NDotV, roughness) * CalculateGeometrySchlickGGX(NDotL, roughness);
	float3 F = CalculateFresnelSchlick(LDotH, reflectance);

	float numerator = NDF * G * F;
	float denominator = 4.0 * NDotV * NDotL + 0.0001;

	return (numerator / denominator) * NDotL * lightColor * attenuation * falloff;
}

half3 CalculateIndirectDiffuse(half3 bakedGI, half occlusion)
{
	return bakedGI;
}

half3 CalculateIndirectSpecular(half3 normalWS, half3 positionWS, half3 viewDirectionWS, half roughness, half occlusion, half3 reflectance)
{
	half3 reflectVector = reflect(-viewDirectionWS, normalWS);
	half NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	half3 envirnonmentReflection = max(0, float3(0, 0, 0));
	half2 fresnelResponse = CalculateFresnelResponse(NDotV, roughness);

	return envirnonmentReflection * (reflectance * fresnelResponse.x + fresnelResponse.y);
}

float3 CalculatePBR(SurfaceData surfaceData, InputData inputData)
{
	float2 diffuseExponent;
	float2 specularExponent;
	float2 specularScale;
	float geometricRoughness = AdjustRoughnessByGeometricNormal(surfaceData.roughness, inputData.normalGS);
	RoughnessEllipseToScaleAndExp(geometricRoughness, diffuseExponent, specularExponent, specularScale);

	float3 reflectance = ((surfaceData.albedo.rgb - 0.04) * surfaceData.metallic + 0.04) * (1 - geometricRoughness);

	float3 directDiffuseTerm = (float3)0;
	float3 directSpecularTerm = (float3)0;

	float3 indirectDiffuseTerm = CalculateIndirectDiffuse(float3(0.01, 0.01, 0.01), surfaceData.occlusion);
	float3 indirectSpecularTerm = CalculateIndirectSpecular(inputData.normalWS, inputData.positionWS, inputData.viewDirectionWS, geometricRoughness, surfaceData.occlusion, reflectance);

	if (_MainShadowCascades[0].w > 0)
	{
		float cascadeIndex = ComputeCascadeIndex(inputData.positionWS, _MainShadowCascades);
		float4 positionSS = TransformWorldToShadow(inputData.positionWS, _MainWorldToShadow[cascadeIndex]);

		//return float4(1 * (cascadeIndex == 0), 1 * (cascadeIndex == 1), 1 * (cascadeIndex == 2), 1);

		float shadowAttenuation = 1;
		if (!IsOutOfBounds(positionSS, _MainShadowBounds[cascadeIndex]))
		{
			shadowAttenuation = ComputeShadowPCF3x3(positionSS, _ShadowTexture, _ShadowTexture_Sampler, _Shadow3x3PCFTermC0, _Shadow3x3PCFTermC1, _Shadow3x3PCFTermC2, _Shadow3x3PCFTermC3);
		}

		float falloff = 1;
		float3 lightDirectionWS = _MainLightDirection.xyz;
		float3 lightColor = _MainLightColor.rgb;

		directDiffuseTerm += CalculateDirectDiffuse(inputData.normalWS, lightDirectionWS, lightColor, shadowAttenuation, falloff, diffuseExponent);
		directSpecularTerm += CalculateDirectSpecular(inputData.normalWS, inputData.viewDirectionWS, lightDirectionWS, lightColor, shadowAttenuation, falloff, reflectance, geometricRoughness);
	}

	for (int i = 0; i < int(_LightsCount.x); i++)
	{
		float4 lightPositionWS = _LightPosition[i];
		float3 posToLight = lightPositionWS.xyz - inputData.positionWS * lightPositionWS.w;
		float distanceSqr = dot(posToLight, posToLight);

		if (distanceSqr > _LightParam[i].w)
		{
			continue;
		}

		float3 lightDirectionWS = normalize(posToLight);
		float spotAttenuation = AngleAttenuation(lightDirectionWS, _LightDirection[i].xyz, _LightAttenuation[i].zw);

		if (spotAttenuation <= 0.0)
		{
			continue;
		}

		float distanceAttenuation = DistanceAttenuation(distanceSqr, _LightAttenuation[i].xy);
		float falloff = LightFalloff(distanceSqr);
		float3 lightColor = _LightColor[i].rgb;

		float shadowAttenuation = 1.0;
		if (_LightParam[i].x > 0)
		{
			shadowAttenuation = 0;
			float4 positionSS = TransformWorldToShadow(inputData.positionWS, _WorldToShadow[i]);
			if (!IsOutOfBounds(positionSS, _ShadowBounds[i]))
			{
				shadowAttenuation = ComputeShadowPCF3x3(positionSS, _ShadowTexture, _ShadowTexture_Sampler, _Shadow3x3PCFTermC0, _Shadow3x3PCFTermC1, _Shadow3x3PCFTermC2, _Shadow3x3PCFTermC3);
			}
		}

		directDiffuseTerm += CalculateDirectDiffuse(inputData.normalWS, lightDirectionWS, lightColor, distanceAttenuation * spotAttenuation * shadowAttenuation, falloff, diffuseExponent);
		directSpecularTerm += CalculateDirectSpecular(inputData.normalWS, inputData.viewDirectionWS, lightDirectionWS, lightColor, distanceAttenuation * spotAttenuation * shadowAttenuation, falloff, reflectance, geometricRoughness);
	}

	directDiffuseTerm *= (1.0 - reflectance);
	indirectDiffuseTerm *= (1.0 - reflectance);

	return ((directDiffuseTerm + indirectDiffuseTerm * surfaceData.occlusion) * surfaceData.albedo + directSpecularTerm + indirectSpecularTerm * surfaceData.occlusion);
}

#endif