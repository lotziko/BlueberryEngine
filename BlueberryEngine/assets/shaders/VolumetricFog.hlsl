#ifndef VOLUMETRIC_FOG_INCLUDED
#define VOLUMETRIC_FOG_INCLUDED

float4 SampleVolumetricFog(float linear01Depth, float2 screenuv)
{
	float3 blueNoise = (SAMPLE_TEXTURE2D(_BlueNoiseLUT, _BlueNoiseLUT_Sampler, screenuv * RENDER_TARGET_SIZE_INV_SIZE.xy / 256).rgb) / 255;

	float z = linear01Depth * _FogNearFarClipPlane.w;
	z = (z - _FogNearFarClipPlane.z) / (1 - _FogNearFarClipPlane.z);
	if (z < 0.0)
		return float4(0, 0, 0, 1);

	float3 uvw = float3(screenuv.x + blueNoise.x, 1 - screenuv.y + blueNoise.y, sqrt(z));
	//uvw.xy += cellNoise(uvw.xy * _Screen_TexelSize.zw) * _ScatterFogVolume_TexelSize.xy * 0.8;
	return SAMPLE_TEXTURE3D_LOD(_VolumetricFogTexture, _VolumetricFogTexture_Sampler, uvw, 0);
}

float4 ApplyVolumetricFog(float4 color, float2 uv, float depth)
{
	float linear01Depth = Linearize01Depth(depth, _CameraNearFarClipPlane.zw);
	float4 fog = SampleVolumetricFog(linear01Depth, uv);
	return float4(color.rgb * fog.a + fog.rgb, color.a);
}

#endif