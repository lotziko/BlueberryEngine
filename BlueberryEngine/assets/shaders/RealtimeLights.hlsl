#ifndef REALTIME_LIGHTS_INCLUDED
#define REALTIME_LIGHTS_INCLUDED

#define CLUSTERS_X 30
#define CLUSTERS_Y 17
#define CLUSTERS_Z 16
#define MAX_LIGHTS 64

uint2 GetCluster(float3 positionVS, float2 normalizedScreenSpaceUV)
{
	uint zTile = uint((log(abs(positionVS.z) / _CameraNearFarClipPlane.x) * CLUSTERS_Z) / log(_CameraNearFarClipPlane.w));
	uint3 tile = uint3(normalizedScreenSpaceUV * uint2(CLUSTERS_X, CLUSTERS_Y), zTile);
	return uint2(tile.x * MAX_LIGHTS, tile.y * CLUSTERS_Z + tile.z);
}

uint2 OffsetCluster(uint2 cluster)
{
	cluster.y += CLUSTERS_Z * CLUSTERS_Y;
	return cluster;
}

float DistanceAttenuation(float distanceSqr, float2 lightDistanceAttenuation)
{
	float factor = distanceSqr * lightDistanceAttenuation.x;
	float smoothFactor = saturate(1.0 - factor * factor);
	return smoothFactor * smoothFactor;
}

float AngleAttenuation(float3 spotDirection, float3 lightDirection, float2 spotAttenuation)
{
	half SdotL = dot(spotDirection, lightDirection);
	half atten = saturate(SdotL * spotAttenuation.x + spotAttenuation.y);
	return atten * atten;
}

float LightFalloff(float distanceSqr, float bias = 0)
{
	return rcp(distanceSqr + bias);
}

float2 GetLightmapChartTexcoord(float3 texcoordLightmap)
{
	float4 offsetScale = _PerLightmapInstanceData[LIGHTMAP_CHART_OFFSET.x > 0 ? LIGHTMAP_CHART_OFFSET.x + texcoordLightmap.z : 0];
	return texcoordLightmap.xy * offsetScale.zw + offsetScale.xy;
}

float3 SampleLightmap(float2 uv)
{
	return SAMPLE_TEXTURE2D_LOD(_LightmapTexture, _LightmapTexture_Sampler, uv, 0).rgb;
}

float3 SampleProbeVolume(float3 positionWS, float3 normalWS, ProbeVolumeData probeVolume)
{
	// TODO sparse bricks
	float3 volumeUvw = ((positionWS + normalWS) - probeVolume.positionMinWS) * probeVolume.invSizeWS;
	float3 gridPos = volumeUvw * float3(probeVolume.topLevelSize);
	uint3 cellPos = min((uint3)floor(gridPos), probeVolume.topLevelSize - 1u);
	float3 localInBrickUvw = saturate(gridPos - float3(cellPos));
	uint cellIndex = cellPos.x + probeVolume.topLevelSize.x * (cellPos.y + probeVolume.topLevelSize.y * cellPos.z);
	uint atlasSlot = probeVolume.probeAtlasOffset + cellIndex;
	uint3 atlasBrickCoord = uint3(atlasSlot % 128, (atlasSlot / 128) % 128, atlasSlot / 16384);
	float3 texelCenter = float3(atlasBrickCoord * 4) + saturate(localInBrickUvw) * 3.0f + 0.5f;
	float3 sampleUvw = texelCenter / float3(512.0f, 512.0f, 40.0f);

	float3 positive = saturate(normalWS);
	float3 negative = saturate(-normalWS);
	positive *= positive;
	negative *= negative;
	float3 result = 0;
	result += _ProbeVolumeTexture0.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * positive.x;
	result += _ProbeVolumeTexture1.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * negative.x;
	result += _ProbeVolumeTexture2.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * positive.y;
	result += _ProbeVolumeTexture3.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * negative.y;
	result += _ProbeVolumeTexture4.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * positive.z;
	result += _ProbeVolumeTexture5.SampleLevel(_ProbeVolumeTexture0_Sampler, sampleUvw, 0) * negative.z;
	return result;
}

#endif