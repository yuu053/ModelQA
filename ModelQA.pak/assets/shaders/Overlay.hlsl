cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gMVP;
    float4 gBaseColorFactor;
    float4 gEmissiveFactorAndNormalStrength;
    float4 gMaterialParams;
    float4 gFlags;
    float4 gFlags2;
    float4 gCameraExposure;
    float4 gLightEnv;
    float4 gTextureCoordSets;
    float4 gLightingModes;
    float4 gStyleParams;
};

struct VSInput
{
    float3 position : POSITION;
    float4 color    : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float4 color    : COLOR;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    output.position = mul(float4(input.position, 1.0f), gMVP);
    output.color = input.color;
    return output;
}

float4 PSMain(VSOutput input) : SV_Target
{
    return input.color;
}
