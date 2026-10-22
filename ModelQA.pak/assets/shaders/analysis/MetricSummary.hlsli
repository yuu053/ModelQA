#ifndef MODELQA_METRIC_SUMMARY_HLSLI
#define MODELQA_METRIC_SUMMARY_HLSLI
// ABI: src/Analysis/GpuAnalysisContract.hpp (32-byte stats, 288-byte summary).
struct MetricStatistics
{
    uint totalCount;
    uint finiteCount;
    uint invalidCount;
    uint skippedCount;
    uint candidateCount;
    float minimum;
    float maximum;
    float average;
};
struct MetricSummary
{
    MetricStatistics statistics;
    uint histogram[64];
};
bool MetricFinite(float value)
{
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}
MetricStatistics MergeStatistics(MetricStatistics a, MetricStatistics b)
{
    MetricStatistics r = (MetricStatistics)0;
    r.totalCount = a.totalCount + b.totalCount;
    r.finiteCount = a.finiteCount + b.finiteCount;
    r.invalidCount = a.invalidCount + b.invalidCount;
    r.skippedCount = a.skippedCount + b.skippedCount;
    r.candidateCount = a.candidateCount + b.candidateCount;
    if (a.finiteCount == 0)
    { r.minimum = b.minimum; r.maximum = b.maximum; r.average = b.average; }
    else if (b.finiteCount == 0)
    { r.minimum = a.minimum; r.maximum = a.maximum; r.average = a.average; }
    else
    {
        r.minimum = min(a.minimum, b.minimum);
        r.maximum = max(a.maximum, b.maximum);
        // Avoid the potentially overflowing sum of N large float errors.
        r.average = a.average + (b.average-a.average) * ((float)b.finiteCount/(float)r.finiteCount);
    }
    return r;
}
#endif
