Texture2D BackgroundTexture : register(t0);
SamplerState BackgroundSampler : register(s0);

struct VSInput { float2 position : POSITION; float2 uv : TEXCOORD0; float4 color : COLOR0; };
struct VSOutput { float4 position : SV_POSITION; float2 uv : TEXCOORD0; float4 color : COLOR0; };

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.position.x * 2.0 - 1.0, 1.0 - input.position.y * 2.0, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

float4 PSColor(VSOutput input) : SV_TARGET { return input.color; }
float4 PSBackground(VSOutput input) : SV_TARGET
{
    return BackgroundTexture.Sample(BackgroundSampler, input.uv) * input.color;
}
