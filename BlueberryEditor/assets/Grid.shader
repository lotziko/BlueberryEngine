struct Attributes
{
	float3 positionOS : POSITION;
	float2 texcoord : TEXCOORD0;
};

struct Varyings
{
	float4 positionCS : SV_POSITION;
	float3 near : TEXCOORD0;
	float3 far : TEXCOORD1;
};

struct Output
{
	float4 color : SV_TARGET;
	float depth : SV_DEPTH;
};

cbuffer PerDrawData : register(b0)
{
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	float4x4 viewProjectionMatrix;
	float4x4 inverseViewMatrix;
	float4x4 inverseProjectionMatrix;
	float4x4 inverseViewProjectionMatrix;
	float4 cameraPositionWS;
};

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/
float3 Unproject(float3 positionCS, float4x4 unprojectMatrix)
{
	float4 unprojectedPoint = mul(float4(positionCS, 1.0f), unprojectMatrix);
	return unprojectedPoint.xyz / unprojectedPoint.w;
}

Varyings Vertex(Attributes input)
{
	Varyings output;
	output.positionCS = float4(input.positionOS, 1.0f);
	output.near = Unproject(float3(input.positionOS.xy, 0.0f), inverseViewProjectionMatrix);
	output.far = Unproject(float3(input.positionOS.xy, 1.0f), inverseViewProjectionMatrix);
	return output;
}

float ComputeDepth(float3 positionWS)
{
	float4 positionCS = mul(float4(positionWS, 1.0f), viewProjectionMatrix);
	return positionCS.z / positionCS.w;
}

// Based on https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
float ComputePristineGrid(float2 uv, float2 lineWidth)
{
	lineWidth = saturate(lineWidth);
	float4 uvDDXY = float4(ddx(uv), ddy(uv));
	float2 uvDeriv = float2(length(uvDDXY.xz), length(uvDDXY.yw));
	bool2 invertLine = lineWidth > 0.5;
	float2 targetWidth = invertLine ? 1.0 - lineWidth : lineWidth;
	float2 drawWidth = clamp(targetWidth, uvDeriv, 0.5);
	float2 lineAA = max(uvDeriv, 0.000001) * 1.5;
	float2 gridUV = abs(frac(uv) * 2.0 - 1.0);
	gridUV = invertLine ? gridUV : 1.0 - gridUV;
	float2 grid2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
	grid2 *= saturate(targetWidth / drawWidth);
	grid2 = lerp(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
	grid2 = invertLine ? 1.0 - grid2 : grid2;
	return lerp(grid2.x, 1.0, grid2.y);
}

Output Fragment(Varyings input)
{
	Output output;

	float t = -input.near.y / (input.far.y - input.near.y);
	clip(t);
	float3 positionWS = input.near + t * (input.far - input.near);
	float gridScale = 1 * 0.1;
	float2 uv = (positionWS * gridScale - floor(cameraPositionWS * gridScale)).xz;

	float gridA = ComputePristineGrid(uv, float2(0.005, 0.005)) * 0.25;
	float gridB = ComputePristineGrid(uv * 10, float2(0.005, 0.005) * 10) * 0.075;
	float depth = ComputeDepth(positionWS);

	output.color = float4(1, 1, 1, max(gridA, gridB));
	output.depth = depth;
	return output;
}
