#ifndef PBR_INCLUDED
#define PBR_INCLUDED

#include "Structs.hlsl"
#include "RealtimeLights.hlsl"
#include "RealtimeShadows.hlsl"

float CalculateGeometricRoughnessFactor(float3 geometricNormalWs)
{
	float3 normalWsDdx = ddx(geometricNormalWs.xyz);
	float3 normalWsDdy = ddy(geometricNormalWs.xyz);
	return pow(saturate(max(dot(normalWsDdx.xyz, normalWsDdx.xyz), dot(normalWsDdy.xyz, normalWsDdy.xyz))), 0.333);
}

float AdjustRoughnessByGeometricNormal(float roughness, float3 geometricNormalWs)
{
	float geometricRoughnessFactor = CalculateGeometricRoughnessFactor(geometricNormalWs.xyz);
	roughness = max(roughness, geometricRoughnessFactor.x);
	return roughness;
}

float2 CalculateFresnelResponse(float NdotV, float roughness)
{
	return pow(SAMPLE_TEXTURE2D(_BRDFIntegrationLUT, _BRDFIntegrationLUT_Sampler, float2(NdotV, roughness)).rg, 2);
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

float CalculateGeometrySchlickGGX(float NDotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float num = NDotV;
	float denom = NDotV * (1.0 - k) + k;

	return num / denom;
}

float3 CalculateFresnelSchlick(float LDotH, float3 reflectance)
{
	return reflectance + (1.0 - reflectance) * pow(1.0 - LDotH, 5);
}

float3 CalculateDirectSpecular(float3 normalWS, float3 viewDirectionWS, float3 lightDirectionWS, float3 lightColor, float attenuation, float falloff, float3 reflectance, float roughness)
{
	float3 floatAngleDirWS = normalize(lightDirectionWS.xyz + viewDirectionWS.xyz);
	float LDotH = max(0, dot(lightDirectionWS.xyz, floatAngleDirWS.xyz));
	float NDotH = max(0, dot(normalWS.xyz, floatAngleDirWS.xyz));
	float NDotL = max(0, dot(normalWS.xyz, lightDirectionWS.xyz));
	float NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	float NDF = CalculateDistributionGGX(NDotH, roughness);
	float G = CalculateGeometrySchlickGGX(NDotV, roughness) * CalculateGeometrySchlickGGX(NDotL, roughness);
	float3 F = CalculateFresnelSchlick(LDotH, reflectance);

	float numerator = NDF * G * F;
	float denominator = 4.0 * NDotV * NDotL + 0.0001;

	return (numerator / denominator) * NDotL * lightColor * attenuation * falloff;
}

float3 CalculateIndirectDiffuse(float3 bakedGI)
{
	return bakedGI;
}

float3 CalculateIndirectSpecular(float3 normalWS, float3 viewDirectionWS, float roughness, float3 reflectance)
{
	float3 reflectVector = reflect(-viewDirectionWS, normalWS);
	float NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	float3 environmentReflection = max(0, SAMPLE_TEXTURECUBE_ARRAY_LOD(_ReflectionTexture, _ReflectionTexture_Sampler, reflectVector, 0, roughness * REFLECTION_LOD_COUNT).rgb);
	float2 fresnelResponse = CalculateFresnelResponse(NDotV, roughness);

	return environmentReflection * (reflectance * fresnelResponse.x + fresnelResponse.y);
}

float3 CalculateIndirectSpecular(float3 normalWS, float3 positionWS, float3 viewDirectionWS, float roughness, float3 reflectance, float3 centerWS, float squareRange, uint reflectionProbeIndex)
{
	float3 reflectVector = reflect(-viewDirectionWS, normalWS);
	float NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	reflectVector = GetSphereProjectionDirection(reflectVector, positionWS, centerWS, squareRange);

	float3 environmentReflection = max(0, SAMPLE_TEXTURECUBE_ARRAY_LOD(_ReflectionTexture, _ReflectionTexture_Sampler, reflectVector, reflectionProbeIndex, roughness * REFLECTION_LOD_COUNT).rgb);
	float2 fresnelResponse = CalculateFresnelResponse(NDotV, roughness);

	return environmentReflection * (reflectance * fresnelResponse.x + fresnelResponse.y);
}

float3 CalculateIndirectSpecular(float3 normalWS, float3 positionWS, float3 viewDirectionWS, float roughness, float3 reflectance, float3 centerWS, float3 minWS, float3 maxWS, uint reflectionProbeIndex)
{
	float3 reflectVector = reflect(-viewDirectionWS, normalWS);
	float NDotV = max(0, dot(normalWS.xyz, viewDirectionWS.xyz));

	reflectVector = GetBoxProjectionDirection(reflectVector, positionWS, centerWS, minWS, maxWS);

	float3 environmentReflection = max(0, SAMPLE_TEXTURECUBE_ARRAY_LOD(_ReflectionTexture, _ReflectionTexture_Sampler, reflectVector, reflectionProbeIndex, roughness * REFLECTION_LOD_COUNT).rgb);
	float2 fresnelResponse = CalculateFresnelResponse(NDotV, roughness);

	return environmentReflection * (reflectance * fresnelResponse.x + fresnelResponse.y);
}

float3 CalculatePBR(SurfaceData surfaceData, InputData inputData)
{
	float2 diffuseExponent;
	float2 specularExponent;
	float2 specularScale;
	float geometricRoughness = AdjustRoughnessByGeometricNormal(surfaceData.roughness, inputData.normalGS);
	RoughnessEllipseToScaleAndExp(geometricRoughness, diffuseExponent, specularExponent, specularScale);

	float3 albedo = (1 - surfaceData.metallic) * surfaceData.albedo;
	float3 reflectance = 0.04;
	reflectance = lerp(reflectance, surfaceData.albedo.rgb, surfaceData.metallic);
	
	float3 directDiffuseTerm = (float3)0;
	float3 directSpecularTerm = (float3)0;

	// Main light
	{
		float cascadeIndex = ComputeCascadeIndex(inputData.positionWS, _MainShadowCascades);
		float4 positionSS = ApplyShadowBias(TransformWorldToShadow(inputData.positionWS, _MainWorldToShadow[cascadeIndex]), inputData.normalWS, _MainLightDirection.xyz);

		float shadowAttenuation = 1;
#if (SHADOWS)
		if (_MainLightHasShadow)
		{
			if (!IsOutOfBounds(positionSS, _MainShadowBounds[cascadeIndex]))
			{
				shadowAttenuation = SampleShadowAtlas(positionSS);
			}
		}
#endif
		float falloff = 1;
		float3 lightDirectionWS = _MainLightDirection.xyz;
		float3 lightColor = _MainLightColor.rgb;

		directDiffuseTerm += CalculateDirectDiffuse(inputData.normalWS, lightDirectionWS, lightColor, shadowAttenuation, falloff, diffuseExponent);
		directSpecularTerm += CalculateDirectSpecular(inputData.normalWS, inputData.viewDirectionWS, lightDirectionWS, lightColor, shadowAttenuation, falloff, reflectance, geometricRoughness);
	}

	uint2 pointCluster = GetCluster(inputData.positionVS, inputData.normalizedScreenSpaceUV);
	uint2 spotCluster = OffsetCluster(pointCluster);
	uint2 reflectionCluster = OffsetCluster(spotCluster);

	// Point lights
	[loop]
	for (int j = 0; j < MAX_LIGHTS; j++)
	{
		uint i = LOAD_TEXTURE2D(_LightIndexTexture, pointCluster).r;
		if (i == 0xFFFF)
		{
			break;
		}
		pointCluster.x += 1;

		PointLightData data = _PointLightsData[i];

		float3 lightPositionWS = data.positionWS;
		float3 posToLight = lightPositionWS - inputData.positionWS;
		float distanceSqr = dot(posToLight, posToLight);

		float3 lightDirectionWS = normalize(posToLight);
		float distanceAttenuation = DistanceAttenuation(distanceSqr, data.attenuation.xy);
		float falloff = LightFalloff(distanceSqr);
		float3 lightColor = data.color;

		float shadowAttenuation = 1.0;
#if (SHADOWS)
		if ((data.flags & 1) != 0)	// has shadow
		{
			uint faceIndex = GetFaceIndex(-lightDirectionWS);
			ShadowData shadowData = _ShadowsData[data.shadowDataOffset + faceIndex];
			shadowAttenuation = 0;
			float4 positionSS = ApplyShadowBias(TransformWorldToShadow(inputData.positionWS, shadowData.worldToShadow));
			if (!IsOutOfBounds(positionSS, shadowData.shadowBounds))
			{
				shadowAttenuation = SampleShadowAtlas(positionSS);
			}
		}
#endif
		directDiffuseTerm += CalculateDirectDiffuse(inputData.normalWS, lightDirectionWS, lightColor, distanceAttenuation * shadowAttenuation, falloff, diffuseExponent);
		directSpecularTerm += CalculateDirectSpecular(inputData.normalWS, inputData.viewDirectionWS, lightDirectionWS, lightColor, distanceAttenuation * shadowAttenuation, falloff, reflectance, geometricRoughness);
	}

	// Spot lights
	[loop]
	for (j = 0; j < MAX_LIGHTS; j++)
	{
		uint i = LOAD_TEXTURE2D(_LightIndexTexture, spotCluster).r;
		if (i == 0xFFFF)
		{
			break;
		}
		spotCluster.x += 1;
		
		SpotLightData data = _SpotLightsData[i];

		float3 lightPositionWS = data.positionWS;
		float3 posToLight = lightPositionWS - inputData.positionWS;
		float distanceSqr = dot(posToLight, posToLight);

		float3 lightDirectionWS = normalize(posToLight);
		float distanceAttenuation = DistanceAttenuation(distanceSqr, data.attenuation.xy);
		float spotAttenuation = AngleAttenuation(lightDirectionWS, data.directionWS, data.attenuation.zw);
		float falloff = LightFalloff(distanceSqr);
		float3 lightColor = data.color;

		float cookieAttenuation = 1.0;
		if ((data.flags & 4) != 0)	// has cookie
		{
			float4 positionLCS = TransformWorldToShadow(inputData.positionWS, data.worldToCookie);
			cookieAttenuation = SAMPLE_TEXTURE3D_LOD(_CookieTexture, _CookieTexture_Sampler, positionLCS.xyz, 0).r;
			cookieAttenuation *= (positionLCS.x > 0.0 && positionLCS.x < 1.0 && positionLCS.y > 0.0 && positionLCS.y < 1.0);
		}

		float shadowAttenuation = 1.0;
#if (SHADOWS)
		if ((data.flags & 1) != 0)	// has shadow
		{
			ShadowData shadowData = _ShadowsData[data.shadowDataOffset];
			shadowAttenuation = 0;
			float4 positionSS = ApplyShadowBias(TransformWorldToShadow(inputData.positionWS, shadowData.worldToShadow));
			if (!IsOutOfBounds(positionSS, shadowData.shadowBounds))
			{
				shadowAttenuation = SampleShadowAtlas(positionSS);
			}
		}
#endif

		directDiffuseTerm += CalculateDirectDiffuse(inputData.normalWS, lightDirectionWS, lightColor, distanceAttenuation * spotAttenuation * cookieAttenuation * shadowAttenuation, falloff, diffuseExponent);
		directSpecularTerm += CalculateDirectSpecular(inputData.normalWS, inputData.viewDirectionWS, lightDirectionWS, lightColor, distanceAttenuation * spotAttenuation * cookieAttenuation * shadowAttenuation, falloff, reflectance, geometricRoughness);
	}

	float3 indirectDiffuseTerm = CalculateIndirectDiffuse(inputData.bakedGI);

#if (REFLECTIONS)
	float3 totalSpecular = 0;
	float totalWeight = 0;
	float maxWeight = 0;

	[loop]
	for (j = 0; j < MAX_LIGHTS; j++)
	{
		uint i = LOAD_TEXTURE2D(_LightIndexTexture, reflectionCluster).r;
		if (i == 0xFFFF)
		{
			break;
		}
		reflectionCluster.x += 1;

		ReflectionProbeData data = _ReflectionProbesData[i];
		float t;
		float3 specular;

		if (data.type == 0)
		{
			float3 posToProbe = inputData.positionWS - data.positionWS;
			float squareDistance = dot(posToProbe, posToProbe);
			float range = data.positionMinWS.x;
			if (squareDistance >= data.squareRange)
			{
				continue;
			}
			t = 1.0f - saturate((sqrt(squareDistance) + data.fade - range) / data.fade);
			specular = CalculateIndirectSpecular(inputData.normalWS, inputData.positionWS, inputData.viewDirectionWS, geometricRoughness, reflectance, data.positionWS, data.squareRange, data.index);
		}
		else if (data.type == 1)
		{
			if (!IsInsideAABB(inputData.positionWS, data.positionMinWS, data.positionMaxWS))
			{
				continue;
			}
			float3 d = max(0, max((data.positionMinWS + data.fade) - inputData.positionWS, inputData.positionWS - (data.positionMaxWS - data.fade)));
			t = 1.0f - saturate(length(d) / data.fade);
			specular = CalculateIndirectSpecular(inputData.normalWS, inputData.positionWS, inputData.viewDirectionWS, geometricRoughness, reflectance, data.positionWS, data.positionMinWS, data.positionMaxWS, data.index);
		}

		float weight = t * t * (3.0 - 2.0 * t);
		maxWeight = max(weight, maxWeight);
		weight *= data.weight;

		totalSpecular += specular * weight;
		totalWeight += weight;
	}

	float3 skySpecular = CalculateIndirectSpecular(inputData.normalWS, inputData.viewDirectionWS, geometricRoughness, reflectance);
	float skyWeight = saturate(1.0f - maxWeight);
	float3 indirectSpecularTerm = lerp(totalSpecular / max(totalWeight, 1e-5), skySpecular, skyWeight);	// TODO sky debug mode by replacing skySpecular with float3(1, 0, 0)
#else
	float3 indirectSpecularTerm = 0;
#endif

	return ((directDiffuseTerm + indirectDiffuseTerm * surfaceData.occlusion) * albedo + directSpecularTerm + indirectSpecularTerm * surfaceData.occlusion);
}

#endif