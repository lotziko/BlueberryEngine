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

float3 SampleLightmap(float2 uv)
{
	return SAMPLE_TEXTURE2D_LOD(_LightmapTexture, _LightmapTexture_Sampler, uv, 0).rgb;
}

static const int3 kVolumeOffsets[8] =
{
	int3(0, 0, 0),
	int3(1, 0, 0),
	int3(0, 1, 0),
	int3(1, 1, 0),
	int3(0, 0, 1),
	int3(1, 0, 1),
	int3(0, 1, 1),
	int3(1, 1, 1),
};

float3 SampleProbeVolume(float3 positionWS, float3 normalWS)
{
	float3 uvw = (positionWS - _ProbeVolumeMin.xyz) * _ProbeVolumeInvSize.xyz;
	float3 baseCoord = uvw * (_ProbeVolumeSize.xyz - 1);
	int3 baseGridCoord = floor(baseCoord);
	float3 frac = baseCoord - baseGridCoord;

	float weightSum = 0;
	float3 colorSum = 0;

	for (int i = 0; i < 8; ++i)
	{
		int3 offset = int3(i, i >> 1, i >> 2) & int3(1, 1, 1);
		int3 probeGridCoord = baseGridCoord + offset;

		float3 probePositionWS = _ProbeVolumeMin.xyz + (float3)offset * _ProbeVolumeCellSize.xyz;
		float3 probeToPos = positionWS - probePositionWS;
		float3 directionWS = normalize(-probeToPos);
		float distance = length(probeToPos);

		float3 trilinear = lerp(1.0 - frac, frac, offset);
		float weight = trilinear.x * trilinear.y * trilinear.z * max(0.005, dot(directionWS, normalWS));

		weightSum += weight;
		colorSum += LOAD_TEXTURE3D(_ProbeVolumeTexture, probeGridCoord).rgb * weight;
	}

	return colorSum / weightSum;
}

#endif