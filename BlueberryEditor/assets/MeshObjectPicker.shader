Shader
{
	Options
	{
		BlendSrc One
		BlendDst Zero
		ZWrite On
		Cull None
	}

	HLSLBEGIN
	#include "Input.hlsl"

	cbuffer PerObjectData
	{
		float4 _ObjectId;
	}

	struct Attributes
	{
		float3 positionOS : POSITION;
	};

	struct Varyings
	{
		float4 positionCS : SV_POSITION;
	};

	Varyings Vertex(Attributes input)
	{
		Varyings output;
		output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), _ViewProjectionMatrix);
		return output;
	}

	float4 Fragment(Varyings input) : SV_TARGET
	{
		return float4(_ObjectId.r, _ObjectId.g, 0, 1);
	}
	HLSLEND
}