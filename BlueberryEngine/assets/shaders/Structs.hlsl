#ifndef STRUCTS_INCLUDED
#define STRUCTS_INCLUDED

struct InputData
{
	float3 positionWS;
	float3 positionVS;
	float3 normalWS;
	float3 normalGS; //geometric roughness
	float3 viewDirectionWS;
	float2 normalizedScreenSpaceUV;
	float3 bakedGI;
};

struct SurfaceData
{
	float3 albedo;
	float alpha;
	float metallic;
	float roughness;
	float3 normalTS;
	float3 emission;
	float occlusion;
};

#endif