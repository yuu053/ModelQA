#include "MetricSummary.hlsli"

StructuredBuffer<float3> Normals : register(t0);
StructuredBuffer<uint> NormalPresent : register(t1);
RWStructuredBuffer<float> ErrorMagnitude : register(u0);
RWStructuredBuffer<uint> Candidates : register(u1);
RWStructuredBuffer<MetricSummary> GroupSummaries : register(u2);
RWByteAddressBuffer CandidateCounters : register(u3); // candidate count, omitted (8 bytes)
cbuffer NormalConstants : register(b0)
{
    uint ElementCount;
    uint CandidateCapacity;
    float Threshold;
    float HistogramUpper;
};
groupshared MetricStatistics SharedStats[256];
groupshared uint SharedHistogram[64];
groupshared uint SharedCandidateCount;
groupshared uint SharedCandidateBase;

[numthreads(1,1,1)]
void CSClear(uint3 tid : SV_DispatchThreadID)
{
    uint bytes; CandidateCounters.GetDimensions(bytes);
    if (tid.x == 0 && tid.y == 0 && tid.z == 0 && bytes >= 8)
    { CandidateCounters.Store(0,0); CandidateCounters.Store(4,0); }
}

// Dispatch (ceil(ElementCount/256),1,1), at most 65535 groups. Padded
// invocations must participate in EVERY barrier; never return before a sync.
[numthreads(256,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane < 64) SharedHistogram[lane] = 0;
    if (lane == 0) { SharedCandidateCount=0; SharedCandidateBase=0xffffffffu; }
    GroupMemoryBarrierWithGroupSync();
    uint nCount, nStride, pCount, pStride, eCount, eStride, cCount, cStride, sCount, sStride, counterBytes;
    Normals.GetDimensions(nCount,nStride);
    NormalPresent.GetDimensions(pCount,pStride);
    ErrorMagnitude.GetDimensions(eCount,eStride);
    Candidates.GetDimensions(cCount,cStride);
    GroupSummaries.GetDimensions(sCount,sStride);
    CandidateCounters.GetDimensions(counterBytes);
    const uint capacity = min(min(CandidateCapacity,cCount),4096u);
    const bool configurationValid = ElementCount <= 65535u*256u &&
        ElementCount <= nCount && ElementCount <= pCount && ElementCount <= eCount &&
        sCount >= ElementCount/256u + (ElementCount%256u != 0 ? 1u : 0u) &&
        nStride == 12 && pStride == 4 && eStride == 4 && cStride == 4 && sStride == 288 &&
        counterBytes >= 8 && MetricFinite(Threshold) && Threshold >= 0 &&
        MetricFinite(HistogramUpper) && HistogramUpper > 0;
    const bool active = tid.x < ElementCount;
    MetricStatistics stats = (MetricStatistics)0;
    float error = asfloat(0x7fc00000u);
    uint localCandidate = 0;
    if (active)
    {
        stats.totalCount = 1;
        bool valid = configurationValid;
        bool present = true;
        if (configurationValid) present = NormalPresent[tid.x] != 0;
        if (!present) stats.skippedCount = 1;
        else
        {
            if (valid)
            {
                const float3 n = Normals[tid.x];
                valid = all((asuint(n) & 0x7f800000u) != 0x7f800000u);
                if (valid)
                {
                    const float scale = max(abs(n.x),max(abs(n.y),abs(n.z)));
                    valid = scale > 0;
                    if (valid)
                    {
                        const float3 scaled = n/scale;
                        precise float lengthValue = scale*sqrt(dot(scaled,scaled));
                        error = abs(lengthValue-1.0f);
                        valid = MetricFinite(error);
                    }
                }
            }
            if (valid)
            {
                stats.finiteCount=1; stats.minimum=error; stats.maximum=error; stats.average=error;
                const uint bin = error >= HistogramUpper ? 63u : min((uint)((error/HistogramUpper)*63.0f),62u);
                InterlockedAdd(SharedHistogram[bin],1);
            }
            else { stats.invalidCount=1; error=asfloat(0x7fc00000u); }
            // Conservative candidate band. These are NOT confirmed defects;
            // threshold-boundary decisions require the CPU reference pass.
            if (!valid || error >= max(0.0f,Threshold-0.00001f*max(error,1.0f)))
            {
                stats.candidateCount=1;
                InterlockedAdd(SharedCandidateCount,1,localCandidate);
            }
        }
        if (tid.x < eCount) ErrorMagnitude[tid.x]=error;
    }
    SharedStats[lane]=stats;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && SharedCandidateCount != 0 && counterBytes >= 8)
    {
        // One global reservation per group, not one atomic per vertex.
        CandidateCounters.InterlockedAdd(0,SharedCandidateCount,SharedCandidateBase);
        const uint available = SharedCandidateBase < capacity ? capacity-SharedCandidateBase : 0;
        const uint omitted = SharedCandidateCount-min(SharedCandidateCount,available);
        if (omitted != 0) { uint ignored; CandidateCounters.InterlockedAdd(4,omitted,ignored); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (stats.candidateCount != 0 && SharedCandidateBase < capacity && localCandidate < capacity-SharedCandidateBase)
        Candidates[SharedCandidateBase+localCandidate]=tid.x;
    for (uint stride=128; stride>0; stride>>=1)
    {
        if (lane < stride) SharedStats[lane]=MergeStatistics(SharedStats[lane],SharedStats[lane+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && group.x < sCount)
    {
        GroupSummaries[group.x].statistics=SharedStats[0];
        for (uint i=0;i<64;++i) GroupSummaries[group.x].histogram[i]=SharedHistogram[i];
    }
}
