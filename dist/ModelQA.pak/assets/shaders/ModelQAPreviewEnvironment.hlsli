// ModelQA Preview Environment 共通関数
// 目的:
//   - Skybox背景、PBR反射、ガラス屈折で同じ方向ベース環境を使う。
//   - 実行時HLSLコンパイルが重くなりすぎないよう、手続き型環境は軽量な式だけにする。
//   - 画像Skyboxがある場合はそちらを優先し、無い場合だけ手続き型へ退避する。
#ifndef MODEL_QA_PREVIEW_ENVIRONMENT_HLSLI
#define MODEL_QA_PREVIEW_ENVIRONMENT_HLSLI

static const float MQA_PI      = 3.14159265359f;
static const float MQA_INV_PI  = 0.31830988618f;
static const float MQA_INV_2PI = 0.15915494309f;

float3 MQA_SafeNormalize(float3 v, float3 fallback)
{
    const float lenSq = dot(v, v);
    return (lenSq > 1.0e-8f) ? v * rsqrt(lenSq) : fallback;
}

float MQA_Luminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 MQA_SRGBToLinear(float3 c)
{
    c = saturate(c);
    return c * c;
}

float3 MQA_LinearToSRGB(float3 c)
{
    return sqrt(saturate(c));
}

float3 MQA_TonemapReinhard(float3 c)
{
    c = max(c, 0.0f);
    return c / (1.0f + c);
}

bool MQA_HasImageSkyboxPreset(float preset)
{
    return preset >= 100.0f;
}

bool MQA_IsHdriSkyboxPreset(float preset)
{
    // base preset 8 = packaged Ferndale Studio 04 HDRI in ModelQA.mqahdri format.
    return MQA_HasImageSkyboxPreset(preset) && abs((preset - 100.0f) - 8.0f) < 0.5f;
}

float MQA_BaseSkyboxPreset(float preset)
{
    return MQA_HasImageSkyboxPreset(preset) ? (preset - 100.0f) : preset;
}

float2 MQA_SkyboxImageUV(float3 dir)
{
    dir = MQA_SafeNormalize(dir, float3(0.0f, 0.0f, 1.0f));
    const float u = atan2(dir.x, dir.z) * MQA_INV_2PI + 0.5f;
    const float v = 0.5f - asin(clamp(dir.y, -1.0f, 1.0f)) * MQA_INV_PI;
    return float2(frac(u), saturate(v));
}

float MQA_Hash21(float2 p)
{
    p = frac(p * float2(127.1f, 311.7f));
    p += dot(p, p + 19.19f);
    return frac(p.x * p.y);
}

float MQA_SoftRect(float2 uv, float2 c, float2 halfSize, float soft)
{
    float dx = abs(frac(uv.x - c.x + 0.5f) - 0.5f);
    float dy = abs(uv.y - c.y);
    float2 d = float2(dx, dy) - halfSize;
    float outside = length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f);
    return 1.0f - smoothstep(0.0f, soft, outside);
}

float MQA_FloorGrid(float3 dir, float scale, float width)
{
    if (dir.y > -0.035f)
        return 0.0f;
    float2 p = dir.xz / max(-dir.y, 0.035f);
    float2 cell = abs(frac(p * scale) - 0.5f);
    float lineMask = 1.0f - smoothstep(width, width + 0.018f, min(cell.x, cell.y));
    float fade = saturate((-dir.y - 0.035f) * 4.0f) * saturate(1.0f - length(p) * 0.030f);
    return lineMask * fade;
}

float3 MQA_ProceduralBase(float preset, float y)
{
    preset = MQA_BaseSkyboxPreset(preset);

    float3 bottom = float3(0.32f, 0.34f, 0.35f);
    float3 mid    = float3(0.58f, 0.61f, 0.64f);
    float3 top    = float3(0.18f, 0.22f, 0.28f);

    if (preset < 0.5f)       { bottom=float3(0.30f,0.30f,0.30f); mid=float3(0.62f,0.64f,0.66f); top=float3(0.18f,0.20f,0.24f); }
    else if (preset < 2.5f)  { bottom=float3(0.36f,0.46f,0.40f); mid=float3(0.70f,0.86f,1.00f); top=float3(0.10f,0.32f,0.82f); }
    else if (preset < 3.5f)  { bottom=float3(0.32f,0.34f,0.35f); mid=float3(0.68f,0.70f,0.72f); top=float3(0.38f,0.42f,0.48f); }
    else if (preset < 5.5f)  { bottom=float3(0.24f,0.20f,0.16f); mid=float3(0.58f,0.50f,0.42f); top=float3(0.13f,0.12f,0.11f); }
    else if (preset < 6.5f)  { bottom=float3(0.13f,0.14f,0.14f); mid=float3(0.46f,0.48f,0.48f); top=float3(0.05f,0.06f,0.07f); }
    else if (preset < 7.5f)  { bottom=float3(0.02f,0.02f,0.03f); mid=float3(0.06f,0.08f,0.12f); top=float3(0.005f,0.006f,0.018f); }
    else                     { bottom=float3(0.08f,0.08f,0.08f); mid=float3(0.36f,0.36f,0.36f); top=float3(0.04f,0.04f,0.05f); }

    float3 c = lerp(bottom, mid, smoothstep(0.0f, 0.55f, y));
    c = lerp(c, top, smoothstep(0.45f, 1.0f, y));
    return c;
}

float3 MQA_EvaluateModelQAPreviewEnvironment(
    float3 worldDir,
    float preset,
    float3 lightDir,
    float envIntensity,
    float lightIntensity,
    float yaw,
    float cloudAmount,
    float contrast)
{
    float s = sin(yaw);
    float c = cos(yaw);
    float3 dir = MQA_SafeNormalize(worldDir, float3(0.0f, 0.0f, 1.0f));
    dir = float3(dir.x * c - dir.z * s, dir.y, dir.x * s + dir.z * c);
    lightDir = MQA_SafeNormalize(lightDir, float3(-0.35f, 0.78f, -0.52f));

    const float2 uv = MQA_SkyboxImageUV(dir);
    const float y = saturate(dir.y * 0.5f + 0.5f);
    const float p = MQA_BaseSkyboxPreset(preset);
    float3 color = MQA_ProceduralBase(preset, y);

    if (p >= 1.0f && p < 3.5f)
    {
        // 青空/屋外/曇り。低コストな雲と太陽だけを入れる。
        float h = MQA_Hash21(floor(uv * float2(44.0f, 16.0f)));
        float cloud = smoothstep(0.63f, 0.90f, h) * smoothstep(0.50f, 0.72f, uv.y) * (1.0f - smoothstep(0.90f, 0.98f, uv.y)) * cloudAmount;
        color = lerp(color, float3(1.04f, 1.08f, 1.10f), cloud * 0.78f);
        float sun = pow(saturate(dot(dir, lightDir)), 160.0f);
        color += float3(1.0f, 0.86f, 0.58f) * sun * max(lightIntensity, 0.0f) * 2.5f;
    }
    else if (p >= 4.0f && p < 5.5f)
    {
        // 室内。窓やランプを小さめに置き、ガラス反射で周囲物が見えるようにする。
        color += float3(1.7f, 1.9f, 2.2f) * MQA_SoftRect(uv, float2(0.22f,0.52f), float2(0.055f,0.13f), 0.028f);
        color += float3(2.2f, 1.25f, 0.48f) * MQA_SoftRect(uv, float2(0.73f,0.66f), float2(0.045f,0.06f), 0.035f);
        color += float3(0.18f,0.14f,0.10f) * MQA_FloorGrid(dir, 2.2f, 0.018f);
    }
    else if (p >= 6.0f && p < 6.5f)
    {
        // ガレージ。蛍光灯ラインと床グリッド。大きな汚い矩形は入れない。
        float strips = 0.0f;
        strips += MQA_SoftRect(uv, float2(0.25f,0.76f), float2(0.11f,0.012f), 0.018f);
        strips += MQA_SoftRect(uv, float2(0.50f,0.78f), float2(0.13f,0.012f), 0.018f);
        strips += MQA_SoftRect(uv, float2(0.75f,0.76f), float2(0.11f,0.012f), 0.018f);
        color += float3(2.0f, 2.1f, 2.0f) * strips;
        color += float3(0.14f,0.16f,0.16f) * MQA_FloorGrid(dir, 4.0f, 0.014f);
    }
    else if (p >= 7.0f && p < 7.5f)
    {
        // 夜景。遠景の窓明かりだけを薄く出す。モデルに黒い斑点が出ないようHDRは控えめ。
        float horizon = smoothstep(0.48f, 0.54f, uv.y) * (1.0f - smoothstep(0.70f, 0.82f, uv.y));
        float2 cell = floor(uv * float2(96.0f, 42.0f));
        float windowOn = step(0.76f, MQA_Hash21(cell));
        color += float3(1.2f, 0.82f, 0.38f) * windowOn * horizon * 1.3f;
    }
    else if (p >= 8.0f)
    {
        float grid = MQA_FloorGrid(dir, 5.5f, 0.012f);
        color += float3(0.26f, 0.26f, 0.26f) * grid;
        color += float3(2.5f,2.5f,2.4f) * MQA_SoftRect(uv, float2(0.75f,0.61f), float2(0.08f,0.14f), 0.025f);
        color += float3(1.6f,0.1f,0.05f) * MQA_SoftRect(uv, float2(0.17f,0.56f), float2(0.03f,0.05f), 0.018f);
        color += float3(0.1f,1.4f,0.12f) * MQA_SoftRect(uv, float2(0.23f,0.56f), float2(0.03f,0.05f), 0.018f);
        color += float3(0.1f,0.22f,1.6f) * MQA_SoftRect(uv, float2(0.29f,0.56f), float2(0.03f,0.05f), 0.018f);
    }

    float luma = MQA_Luminance(color);
    color = lerp(float3(luma, luma, luma), color, max(contrast, 0.0f));
    return max(color * max(envIntensity, 0.0f), 0.0f);
}

float3 MQA_SampleProceduralEnvironmentRough(
    float3 dir,
    float roughness,
    float preset,
    float3 lightDir,
    float envIntensity,
    float lightIntensity,
    float yaw,
    float cloudAmount,
    float contrast)
{
    float3 c = MQA_EvaluateModelQAPreviewEnvironment(dir, preset, lightDir, envIntensity, lightIntensity, yaw, cloudAmount, contrast);
    float r = saturate(roughness);
    if (r > 0.15f)
    {
        float3 grey = float3(MQA_Luminance(c), MQA_Luminance(c), MQA_Luminance(c));
        c = lerp(c, grey, r * 0.32f);
    }
    return c;
}

#endif
