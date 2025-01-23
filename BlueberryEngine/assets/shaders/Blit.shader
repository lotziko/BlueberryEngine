Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BlitVertex
		#pragma fragment BlitFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
		};

		TEXTURE2D(_BlitTexture);	SAMPLER(_BlitTexture_Sampler);

		Varyings BlitVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		float4 BlitFragment(Varyings input) : SV_TARGET
		{
			return SAMPLE_TEXTURE2D(_BlitTexture, _BlitTexture_Sampler, input.texcoord);
		}
		HLSLEND
	}
}