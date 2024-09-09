#ifndef REALTIME_LIGHTS_INCLUDED
#define REALTIME_LIGHTS_INCLUDED

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