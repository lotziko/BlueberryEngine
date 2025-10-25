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

#endif