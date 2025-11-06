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
			uint instanceID : SV_InstanceID;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			uint renderTargetIndex : SV_RenderTargetArrayIndex;
		};

		Varyings BlitVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			output.renderTargetIndex = input.instanceID;
			return output;
		}

		TEXTURECUBE(_BlitTexture);	SAMPLER(_BlitTexture_Sampler);

		float3 GetDirection(uint face, float2 uv)
		{
			float3 direction[6] =
			{
				float3(1.0, -uv.y, -uv.x),
				float3(-1.0, -uv.y, uv.x),
				float3(uv.x, 1.0, uv.y),
				float3(uv.x, -1.0, -uv.y),
				float3(uv.x, -uv.y, 1.0),
				float3(-uv.x, -uv.y, -1.0)
			};
			return normalize(direction[face]);
		}

		float4 BlitFragment(Varyings input) : SV_TARGET
		{
			return SAMPLE_TEXTURECUBE(_BlitTexture, _BlitTexture_Sampler, GetDirection(input.renderTargetIndex, input.texcoord * 2.0 - 1.0));
		}
		HLSLEND
	}
}