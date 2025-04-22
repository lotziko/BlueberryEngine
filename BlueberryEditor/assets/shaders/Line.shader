Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), VIEW_PROJECTION_MATRIX);
			output.color = input.color;
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		ZTest Greater
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), VIEW_PROJECTION_MATRIX);
			output.color = input.color;
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			input.color.rgb *= 0.25;
			return input.color;
		}
		HLSLEND
	}
}