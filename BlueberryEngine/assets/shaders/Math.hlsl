#ifndef MATH_INCLUDED
#define MATH_INCLUDED

#define PI 3.14159265358979323846
#define INV_PI 1.0 / PI

float3 NormalTSToNormalWS(float3 normalTS, float3 normalWS, float3 tangentWS, float3 bitangentWS)
{
	float3 normal;
	normal.xyz = normalTS.x * tangentWS.xyz;
	normal.xyz += normalTS.y * bitangentWS.xyz;
	normal.xyz += normalTS.z * normalWS.xyz;
	return normalize(normal);
}

float Linearize01Depth(float depth, float2 params)
{
	return 1.0 / (params.x * depth + params.y);
}

float3 ReconstructNormal(float3 normal)
{
	return normalize(float3(normal.x, normal.y, sqrt(saturate(1 - dot(normal.xy, normal.xy)))));
}

float2 EncodeNormalOctahedral(float3 normal)
{
	normal /= (abs(normal.x) + abs(normal.y) + abs(normal.z));
	if (normal.z < 0.0)
	{
		normal.xy = (1.0 - abs(normal.yx)) * sign(normal.xy);
	}
	return normal.xy * 0.5 + 0.5;
}

float3 DecodeNormalOctahedral(float2 normal)
{
	normal = normal * 2.0 - 1.0;
	float3 result = float3(normal.x, normal.y, 1.0 - abs(normal.x) - abs(normal.y));
	if (result.z < 0.0)
	{
		result.xy = (1.0 - abs(result.yx)) * sign(result.xy);
	}
	return normalize(result);
}

bool IsInsideAABB(float3 position, float3 min, float3 max)
{
	return (position.x > min.x && position.x < max.x && position.y > min.y && position.y < max.y && position.z > min.z && position.z < max.z);
}

float4 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
	float a = roughness * roughness;

	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float sinTheta = sqrt(max(1e-5, 1.0 - cosTheta * cosTheta));

	// from spherical coordinates to cartesian coordinates
	float3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// pdf
	float d = (cosTheta * (a * a) - cosTheta) * cosTheta + 1;
	float D = (a * a) / (PI * d * d);
	float pdf = D * cosTheta;

	// from tangent-space vector to world-space sample vector
	float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(up, N));
	float3 bitangent = cross(N, tangent);

	float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return float4(normalize(sampleVec), pdf);
}

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

float3 GetSphereProjectionDirection(float3 reflectionWS, float3 positionWS, float3 centerWS, float squareRange)
{
	float3 worldPos = positionWS - centerWS;
	float B = dot(worldPos, reflectionWS);
	float C = dot(worldPos, worldPos) - squareRange;
	float D = B * B - C;
	float t = -B + sqrt(max(D, 0.0));
	return worldPos + reflectionWS * t;
}

float3 GetBoxProjectionDirection(float3 reflectionWS, float3 positionWS, float3 boxCenterWS, float3 boxMinWS, float3 boxMaxWS)
{
	float3 boxMinMax = float3(reflectionWS.x > 0.0f ? boxMaxWS.x : boxMinWS.x, reflectionWS.y > 0.0f ? boxMaxWS.y : boxMinWS.y, reflectionWS.z > 0.0f ? boxMaxWS.z : boxMinWS.z);
	float3 rbMinMax = (boxMinMax - positionWS) / reflectionWS;
	float fa = min(min(rbMinMax.x, rbMinMax.y), rbMinMax.z);
	float3 worldPos = positionWS - boxCenterWS;
	return worldPos + reflectionWS * fa;
}

#endif