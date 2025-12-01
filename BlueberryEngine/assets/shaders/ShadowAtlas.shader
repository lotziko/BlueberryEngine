Shader
{
	Pass
	{
		Blend One Zero
		Cull None
		ZTest Always

		HLSLBEGIN
		#pragma vertex ClearVertex
		#pragma fragment ClearFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
		};

		Varyings ClearVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			return output;
		}

		float ClearFragment(Varyings input) : SV_DEPTH
		{
			return 1;
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
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

		cbuffer DepthBlitData
		{
			float4 _OffsetScale;
		};

		TEXTURE2D(_BlitTexture);	SAMPLER(_BlitTexture_Sampler);

		Varyings BlitVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		float BlitFragment(Varyings input) : SV_DEPTH
		{
			return SAMPLE_TEXTURE2D(_BlitTexture, _BlitTexture_Sampler, input.texcoord * _OffsetScale.zw + _OffsetScale.xy).r;
		}
		HLSLEND
	}
}