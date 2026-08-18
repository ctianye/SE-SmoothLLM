# Data health report

This report is generated before plotting. It checks the cleaned metric tables, not the raw response text.

## raw_metrics

- Rows: 1,800
- Columns: 17
- Memory: 0.95 MB
- Exact duplicate rows: 0

### Column inventory

```text
             column   dtype  missing  missing_pct  n_unique
             method  object        0          0.0         3
               seed   int64        0          0.0         3
              split  object        0          0.0         2
              index   int64        0          0.0       100
           behavior  object        0          0.0       100
           category  object        0          0.0        10
      config_sha256  object        0          0.0         1
              model  object        0          0.0         1
internal_jailbroken    bool        0          0.0         2
        copies_used   int64        0          0.0         7
      stopped_early    bool        0          0.0         2
      prompt_tokens   int64        0          0.0       644
  completion_tokens   int64        0          0.0       684
         latency_ms float64        0          0.0      1800
         votes_safe   int64        0          0.0        11
   votes_jailbroken   int64        0          0.0        11
       trace_length   int64        0          0.0         7
```

### Numeric summaries

```text
 count      mean       std      min       25%       50%       75%        max
1800.0    43.000     0.817   42.000    42.000    43.000    44.000     44.000
1800.0    49.500    28.874    0.000    24.750    49.500    74.250     99.000
1800.0     5.858     3.802    1.000     1.000     6.000    10.000     10.000
1800.0   489.049   359.408   47.000    77.000   514.500   703.250   1286.000
1800.0   696.001   473.350   26.000   150.000   746.000  1069.750   1500.000
1800.0 49455.321 33547.887 1950.138 10592.199 53303.163 76270.827 106908.961
1800.0     2.887     3.272    0.000     0.000     1.000     5.000     10.000
1800.0     2.972     3.040    0.000     1.000     1.000     6.000     10.000
1800.0     5.858     3.802    1.000     1.000     6.000    10.000     10.000
```

### Normality checks (descriptive only)

```text
           column  shapiro_p    verdict
             seed        0.0 not normal
            index        0.0 not normal
      copies_used        0.0 not normal
    prompt_tokens        0.0 not normal
completion_tokens        0.0 not normal
       latency_ms        0.0 not normal
       votes_safe        0.0 not normal
 votes_jailbroken        0.0 not normal
     trace_length        0.0 not normal
```

### Outlier checks (IQR rule)

```text
           column  iqr_outliers  outlier_pct
             seed             0          0.0
            index             0          0.0
      copies_used             0          0.0
    prompt_tokens             0          0.0
completion_tokens             0          0.0
       latency_ms             0          0.0
       votes_safe             0          0.0
 votes_jailbroken             0          0.0
     trace_length             0          0.0
```

### Category counts

- `method` unique values: 3
```text
         method  count
   se_smoothllm    600
smoothllm_fixed    600
     undefended    600
```

- `split` unique values: 2
```text
  split  count
 benign    900
harmful    900
```

- `behavior` unique values: 100
```text
          behavior  count
        Defamation     18
    Date rape drug     18
          Tax scam     18
Facial recognition     18
         Blackmail     18
    Identity theft     18
        Pedophilia     18
      Revenge porn     18
 Human trafficking     18
        Bestiality     18
```

- `category` unique values: 10
```text
                  category  count
 Harassment/Discrimination    180
           Malware/Hacking    180
             Physical harm    180
             Economic harm    180
           Fraud/Deception    180
            Disinformation    180
      Sexual/Adult content    180
                   Privacy    180
             Expert advice    180
Government decision-making    180
```

- `config_sha256` unique values: 1
```text
                                                   config_sha256  count
52472121f2a9cfce3d20b810c745a850bb616794a8378065899684fa830126b7   1800
```

- `model` unique values: 1
```text
                model  count
lmsys/vicuna-13b-v1.5   1800
```

## deepseek_judge

- Rows: 600
- Columns: 13
- Memory: 0.32 MB
- Exact duplicate rows: 0

### Column inventory

```text
             column   dtype  missing  missing_pct  n_unique
             method  object        0          0.0         2
               seed   int64        0          0.0         3
              split  object        0          0.0         1
              index   int64        0          0.0       100
    response_sha256  object        0          0.0       461
      config_sha256  object        0          0.0         1
external_jailbroken    bool        0          0.0         2
  format_conforming    bool        0          0.0         1
         raw_output  object        0          0.0         2
              model  object        0          0.0         1
         latency_ms float64        0          0.0       600
      prompt_tokens   int64        0          0.0       120
  completion_tokens   int64        0          0.0         2
```

### Numeric summaries

```text
 count     mean     std      min      25%      50%     75%      max
 600.0   43.000   0.817   42.000   42.000   43.000   44.00   44.000
 600.0   49.500  28.890    0.000   24.750   49.500   74.25   99.000
 600.0 1576.699 396.052 1059.831 1364.578 1477.121 1650.27 4715.305
 600.0  315.675  35.253  246.000  291.000  313.000  349.00  385.000
 600.0    1.062   0.241    1.000    1.000    1.000    1.00    2.000
```

### Normality checks (descriptive only)

```text
           column  shapiro_p    verdict
             seed        0.0 not normal
            index        0.0 not normal
       latency_ms        0.0 not normal
    prompt_tokens        0.0 not normal
completion_tokens        0.0 not normal
```

### Outlier checks (IQR rule)

```text
           column  iqr_outliers  outlier_pct
             seed             0        0.000
            index             0        0.000
       latency_ms            48        8.000
    prompt_tokens             0        0.000
completion_tokens            37        6.167
```

### Category counts

- `method` unique values: 2
```text
         method  count
   se_smoothllm    300
smoothllm_fixed    300
```

- `split` unique values: 1
```text
  split  count
harmful    600
```

- `response_sha256` unique values: 461
```text
                                                 response_sha256  count
a9c4f96378030094282b7c9fadb58e3281ba368d3a3b3759425a4e0311b4eb07     10
6edb7bd630bb2319d659fd77ee11f25d3d1671b80712e4356d5599d98815b7a2      9
0fe0d01671b89ca9b791ef62234e27fc9e8547e814eace6d73f3316c1a6d31ee      9
4f8b8f72de9f6be1b783d48b5770372b5a0850e811ed1d1c8112fda43f104632      7
6e130b5b942eca35c30d3523da60cdf1fd153e2c7ad008466f616ad8ced519c8      7
4cad2f6da0e6b6d4bf93d8ce5b9e6f3bd5aa9507f13cf13c94176ad8b26ddf1c      5
047d4057634d3f02db2311e969313bef63549de4fb45826def6f823e126ff53f      5
86eb125860c3d34cfa634a5561dbd17d6592df56e90c2a175643f7d1c932ad94      4
fb05ce47e57d8979cdbfed02d28e01ea5481caed3858466acb3bbc57c8a391cb      4
bf1b4e4c65eae806c2fe3751c88acd2d5c821ac66154ab66f60a64beeaff1c82      3
```

- `config_sha256` unique values: 1
```text
                                                   config_sha256  count
52472121f2a9cfce3d20b810c745a850bb616794a8378065899684fa830126b7    600
```

- `raw_output` unique values: 2
```text
raw_output  count
      safe    563
    unsafe     37
```

- `model` unique values: 1
```text
            model  count
deepseek-v4-flash    600
```

## refusal_judge

- Rows: 1,800
- Columns: 13
- Memory: 0.97 MB
- Exact duplicate rows: 0

### Column inventory

```text
           column   dtype  missing  missing_pct  n_unique
           method  object        0          0.0         3
             seed   int64        0          0.0         3
            split  object        0          0.0         2
            index   int64        0          0.0       100
  response_sha256  object        0          0.0      1154
    config_sha256  object        0          0.0         1
          refused    bool        0          0.0         2
format_conforming    bool        0          0.0         2
       raw_output  object        0          0.0        22
            model  object        0          0.0         1
       latency_ms float64        0          0.0      1800
    prompt_tokens   int64        0          0.0       133
completion_tokens   int64        0          0.0         3
```

### Numeric summaries

```text
 count    mean     std     min     25%     50%     75%      max
1800.0  43.000   0.817  42.000  42.000  43.000  44.000   44.000
1800.0  49.500  28.874   0.000  24.750  49.500  74.250   99.000
1800.0 355.640 128.341 143.678 250.038 355.224 424.222 1084.664
1800.0 198.628  32.037 105.000 176.000 213.000 221.000  246.000
1800.0   2.992   1.469   2.000   2.000   3.000   3.000    8.000
```

### Normality checks (descriptive only)

```text
           column  shapiro_p    verdict
             seed        0.0 not normal
            index        0.0 not normal
       latency_ms        0.0 not normal
    prompt_tokens        0.0 not normal
completion_tokens        0.0 not normal
```

### Outlier checks (IQR rule)

```text
           column  iqr_outliers  outlier_pct
             seed             0        0.000
            index             0        0.000
       latency_ms            45        2.500
    prompt_tokens             8        0.444
completion_tokens           129        7.167
```

### Category counts

- `method` unique values: 3
```text
         method  count
   se_smoothllm    600
smoothllm_fixed    600
     undefended    600
```

- `split` unique values: 2
```text
  split  count
 benign    900
harmful    900
```

- `response_sha256` unique values: 1154
```text
                                                 response_sha256  count
a9c4f96378030094282b7c9fadb58e3281ba368d3a3b3759425a4e0311b4eb07     10
6edb7bd630bb2319d659fd77ee11f25d3d1671b80712e4356d5599d98815b7a2      9
0fe0d01671b89ca9b791ef62234e27fc9e8547e814eace6d73f3316c1a6d31ee      9
4f8b8f72de9f6be1b783d48b5770372b5a0850e811ed1d1c8112fda43f104632      7
6e130b5b942eca35c30d3523da60cdf1fd153e2c7ad008466f616ad8ced519c8      7
4cad2f6da0e6b6d4bf93d8ce5b9e6f3bd5aa9507f13cf13c94176ad8b26ddf1c      5
047d4057634d3f02db2311e969313bef63549de4fb45826def6f823e126ff53f      5
fb05ce47e57d8979cdbfed02d28e01ea5481caed3858466acb3bbc57c8a391cb      4
86eb125860c3d34cfa634a5561dbd17d6592df56e90c2a175643f7d1c932ad94      4
61f5915ab228d810c8bf76f0cf76f4fa4d425786321e856c86ae0b3ed2e5d305      3
```

- `config_sha256` unique values: 1
```text
                                                   config_sha256  count
52472121f2a9cfce3d20b810c745a850bb616794a8378065899684fa830126b7   1800
```

- `raw_output` unique values: 22
```text
                                         raw_output  count
                                                No.   1012
                                                Yes    601
                                                 No     58
          No.\n\nThe language model does not refuse     32
             No. The language model does not refuse     25
       I cannot provide a response that promotes or     12
I cannot provide information or guidance on illegal     12
             No.\n\nThe response does not contain a      8
                No. The response does not refuse to      4
       I cannot provide instructions on how to make      4
```

- `model` unique values: 1
```text
                        model  count
meta-llama/Llama-3-8b-chat-hf   1800
```

## paired_metrics

- Rows: 600
- Columns: 19
- Memory: 0.10 MB
- Exact duplicate rows: 0

### Column inventory

```text
                    column   dtype  missing  missing_pct  n_unique
                      seed   int64        0          0.0         3
                     split  object        0          0.0         2
                     index   int64        0          0.0       100
 fixed_internal_jailbroken    bool        0          0.0         2
         fixed_copies_used   int64        0          0.0         1
       fixed_prompt_tokens   int64        0          0.0       350
   fixed_completion_tokens   int64        0          0.0       387
          fixed_latency_ms float64        0          0.0       600
       fixed_stopped_early    bool        0          0.0         1
    se_internal_jailbroken    bool        0          0.0         2
            se_copies_used   int64        0          0.0         6
          se_prompt_tokens   int64        0          0.0       356
      se_completion_tokens   int64        0          0.0       390
             se_latency_ms float64        0          0.0       600
          se_stopped_early    bool        0          0.0         2
    internal_verdict_match    bool        0          0.0         1
           query_reduction float64        0          0.0         6
    prompt_token_reduction float64        0          0.0       552
completion_token_reduction float64        0          0.0       484
```

### Numeric summaries

```text
 count      mean       std       min       25%       50%       75%        max
 600.0    43.000     0.817    42.000    42.000    43.000    44.000     44.000
 600.0    49.500    28.890     0.000    24.750    49.500    74.250     99.000
 600.0    10.000     0.000    10.000    10.000    10.000    10.000     10.000
 600.0   849.967   217.079   509.000   644.750   828.500  1045.250   1286.000
 600.0  1165.388   263.541   423.000   963.500  1186.500  1398.000   1500.000
 600.0 82875.369 18233.227 31190.139 68938.441 84225.011 98973.692 106908.961
 600.0     6.575     1.441     5.000     5.000     6.000     7.000     10.000
 600.0   550.980   165.098   271.000   424.000   515.000   622.250   1181.000
 600.0   776.620   270.101   180.000   535.750   845.000   948.000   1500.000
 600.0 55189.980 18943.144 13328.237 38431.984 59758.426 67313.574 106046.387
 600.0     0.342     0.144     0.000     0.300     0.400     0.500      0.500
 600.0     0.342     0.144     0.000     0.292     0.397     0.491      0.509
 600.0     0.343     0.154    -0.002     0.247     0.378     0.445      0.698
```

### Normality checks (descriptive only)

```text
                    column  shapiro_p                  verdict
                      seed        0.0               not normal
                     index        0.0               not normal
         fixed_copies_used        NaN insufficient or constant
       fixed_prompt_tokens        0.0               not normal
   fixed_completion_tokens        0.0               not normal
          fixed_latency_ms        0.0               not normal
            se_copies_used        0.0               not normal
          se_prompt_tokens        0.0               not normal
      se_completion_tokens        0.0               not normal
             se_latency_ms        0.0               not normal
           query_reduction        0.0               not normal
    prompt_token_reduction        0.0               not normal
completion_token_reduction        0.0               not normal
```

### Outlier checks (IQR rule)

```text
                    column  iqr_outliers  outlier_pct
                      seed             0        0.000
                     index             0        0.000
         fixed_copies_used             0        0.000
       fixed_prompt_tokens             0        0.000
   fixed_completion_tokens             0        0.000
          fixed_latency_ms             0        0.000
            se_copies_used             0        0.000
          se_prompt_tokens            25        4.167
      se_completion_tokens             0        0.000
             se_latency_ms             0        0.000
           query_reduction            28        4.667
    prompt_token_reduction             0        0.000
completion_token_reduction             0        0.000
```

### Category counts

- `split` unique values: 2
```text
  split  count
 benign    300
harmful    300
```

## Interpretation

All expected key columns are complete and unique. The numeric distributions are used for descriptive summaries; no inferential significance claims are made because the main comparison has only three configured seeds.
