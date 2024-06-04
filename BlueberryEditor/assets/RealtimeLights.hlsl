#ifndef REALTIME_LIGHTS_INCLUDED
#define REALTIME_LIGHTS_INCLUDED

float DistanceAttenuation(float distanceSqr, float2 lightDistanceAttenuation)
{
	float factor = distanceSqr * lightDistanceAttenuation.x;
	float smoothFactor = saturate(1.0 - factor * factor);
	return smoothFactor * smoothFactor;
}

float LightFalloff(float distanceSqr, float bias = 0)
{
	return rcp(distanceSqr + bias);
}

#endif