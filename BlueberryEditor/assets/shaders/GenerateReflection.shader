Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex GenerateReflectionVertex
		#pragma fragment GenerateReflectionFragment

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

		Varyings GenerateReflectionVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			output.renderTargetIndex = input.instanceID;
			return output;
		}

		TEXTURECUBE(_SourceTexture);	SAMPLER(_SourceTexture_Sampler);

		float RadicalInverse_VdC(uint bits)
		{
			bits = (bits << 16u) | (bits >> 16u);
			bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
			bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
			bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
			bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
			return float(bits) * 2.3283064365386963e-10; // / 0x100000000
		}

		float2 Hammersley(uint i, uint N)
		{
			return float2(float(i) / float(N), RadicalInverse_VdC(i));
		}

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

		#define SAMPLE_COUNT 4096u

		cbuffer _ReflectionGenerationData
		{
			float4 _Roughness;
		};

		float4 GenerateReflectionFragment(Varyings input) : SV_TARGET
		{
			float3 N = GetDirection(input.renderTargetIndex, input.texcoord * 2.0 - 1.0);
			float3 R = N;
			float3 V = R;

			float totalWeight = 0.0;
			float3 prefilteredColor = 0;

			for (uint i = 0u; i < SAMPLE_COUNT; ++i)
			{
				float2 Xi = Hammersley(i, SAMPLE_COUNT);
				float4 H = ImportanceSampleGGX(Xi, N, _Roughness.r);
				float3 L = normalize(2.0 * dot(V, H.xyz) * H.xyz - V);

				float NdotL = max(dot(N, L), 0.0);
				if (NdotL > 0.0)
				{
					prefilteredColor += saturate(SAMPLE_TEXTURECUBE(_SourceTexture, _SourceTexture_Sampler, L).rgb) * NdotL;
					totalWeight += NdotL;
				}
			}
			prefilteredColor = prefilteredColor / totalWeight;
			return float4(prefilteredColor, 1.0);
		}
		HLSLEND
	}
}