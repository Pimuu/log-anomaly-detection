# Results summary (5 seed(s): [0, 1, 2, 3, 4])

Test period only; threshold = 2 % alert budget on the calibration period.

## Overall metrics (mean ± std over seeds)

| method | roc_auc | avg_precision | precision | recall | f1 | best_f1 | alert_rate |
|---|---|---|---|---|---|---|---|
| IsolationForest | 0.978 ± 0.009 | 0.737 ± 0.102 | 0.775 ± 0.110 | 0.594 ± 0.162 | 0.655 ± 0.102 | 0.674 ± 0.101 | 0.016 ± 0.008 |
| LOF | 0.810 ± 0.089 | 0.141 ± 0.068 | 0.209 ± 0.104 | 0.184 ± 0.108 | 0.192 ± 0.102 | 0.244 ± 0.091 | 0.017 ± 0.006 |
| Autoencoder | 0.945 ± 0.021 | 0.348 ± 0.104 | 0.387 ± 0.082 | 0.323 ± 0.116 | 0.344 ± 0.103 | 0.439 ± 0.088 | 0.017 ± 0.007 |
| Ensemble-mean | 0.962 ± 0.014 | 0.488 ± 0.136 | 0.534 ± 0.087 | 0.404 ± 0.150 | 0.448 ± 0.124 | 0.492 ± 0.086 | 0.015 ± 0.006 |
| Ensemble-max | 0.969 ± 0.007 | 0.435 ± 0.097 | 0.541 ± 0.073 | 0.466 ± 0.184 | 0.487 ± 0.138 | 0.548 ± 0.085 | 0.017 ± 0.008 |
| Rules | 0.773 ± 0.054 | 0.462 ± 0.133 | 0.592 ± 0.142 | 0.551 ± 0.107 | 0.570 ± 0.122 | 0.570 ± 0.122 | 0.018 ± 0.003 |

## Dataset statistics

| seed | events | anomalous_events | anomaly_rate | campaigns | fit_events | calib_events | test_events | calib_anomaly_rate | test_anomaly_rate | test_campaigns | fit_seconds_IsolationForest | fit_seconds_LOF | fit_seconds_Autoencoder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 166245.000 | 3474.000 | 0.021 | 92.000 | 54860.000 | 27274.000 | 55788.000 | 0.014 | 0.024 | 34.000 | 3.074 | 9.508 | 15.946 |
| 1.000 | 169922.000 | 3849.000 | 0.023 | 92.000 | 54700.000 | 29344.000 | 57800.000 | 0.031 | 0.014 | 27.000 | 3.116 | 8.368 | 17.921 |
| 2.000 | 177994.000 | 3982.000 | 0.022 | 92.000 | 58699.000 | 30148.000 | 58885.000 | 0.022 | 0.017 | 27.000 | 3.020 | 7.924 | 17.498 |
| 3.000 | 176511.000 | 3627.000 | 0.021 | 92.000 | 59942.000 | 29809.000 | 58379.000 | 0.020 | 0.024 | 35.000 | 3.294 | 8.832 | 13.481 |
| 4.000 | 178391.000 | 3563.000 | 0.020 | 92.000 | 60077.000 | 29883.000 | 59234.000 | 0.031 | 0.018 | 31.000 | 3.633 | 10.151 | 22.705 |

## Per attack type (ROC-AUC & event recall: mean over seeds; campaign detection: pooled over seeds)

| method | attack_type | type_auc | event_recall | detected_campaigns | n_campaigns | campaign_detection_pooled |
|---|---|---|---|---|---|---|
| IsolationForest | brute_force | 1.000 | 1.000 | 22.000 | 22 | 1.000 |
| IsolationForest | odd_hour_login | 0.954 | 0.038 | 19.000 | 50 | 0.380 |
| IsolationForest | password_spray | 1.000 | 1.000 | 13.000 | 13 | 1.000 |
| IsolationForest | privilege_escalation | 0.953 | 0.476 | 23.000 | 30 | 0.767 |
| IsolationForest | rare_ip_takeover | 0.958 | 0.168 | 28.000 | 39 | 0.718 |
| LOF | brute_force | 0.844 | 0.256 | 18.000 | 22 | 0.818 |
| LOF | odd_hour_login | 0.916 | 0.206 | 33.000 | 50 | 0.660 |
| LOF | password_spray | 0.730 | 0.244 | 13.000 | 13 | 1.000 |
| LOF | privilege_escalation | 0.617 | 0.220 | 22.000 | 30 | 0.733 |
| LOF | rare_ip_takeover | 0.831 | 0.041 | 16.000 | 39 | 0.410 |
| Autoencoder | brute_force | 0.985 | 0.360 | 18.000 | 22 | 0.818 |
| Autoencoder | odd_hour_login | 0.937 | 0.082 | 19.000 | 50 | 0.380 |
| Autoencoder | password_spray | 0.991 | 0.612 | 13.000 | 13 | 1.000 |
| Autoencoder | privilege_escalation | 0.946 | 0.546 | 28.000 | 30 | 0.933 |
| Autoencoder | rare_ip_takeover | 0.864 | 0.149 | 23.000 | 39 | 0.590 |
| Ensemble-mean | brute_force | 0.980 | 0.570 | 22.000 | 22 | 1.000 |
| Ensemble-mean | odd_hour_login | 0.982 | 0.296 | 44.000 | 50 | 0.880 |
| Ensemble-mean | password_spray | 0.961 | 0.429 | 13.000 | 13 | 1.000 |
| Ensemble-mean | privilege_escalation | 0.906 | 0.395 | 26.000 | 30 | 0.867 |
| Ensemble-mean | rare_ip_takeover | 0.950 | 0.232 | 29.000 | 39 | 0.744 |
| Ensemble-max | brute_force | 0.995 | 0.851 | 22.000 | 22 | 1.000 |
| Ensemble-max | odd_hour_login | 0.949 | 0.144 | 21.000 | 50 | 0.420 |
| Ensemble-max | password_spray | 0.995 | 0.640 | 13.000 | 13 | 1.000 |
| Ensemble-max | privilege_escalation | 0.944 | 0.381 | 26.000 | 30 | 0.867 |
| Ensemble-max | rare_ip_takeover | 0.943 | 0.083 | 17.000 | 39 | 0.436 |
| Rules | brute_force | 0.976 | 0.953 | 22.000 | 22 | 1.000 |
| Rules | odd_hour_login | 0.510 | 0.028 | 28.000 | 50 | 0.560 |
| Rules | password_spray | 0.892 | 0.791 | 13.000 | 13 | 1.000 |
| Rules | privilege_escalation | 0.874 | 0.754 | 30.000 | 30 | 1.000 |
| Rules | rare_ip_takeover | 0.502 | 0.011 | 14.000 | 39 | 0.359 |

## False-positive sources (summed over seeds)

| method | source | n_events | false_positives | fp_rate |
|---|---|---|---|---|
| IsolationForest | forgot_password | 289 | 149 | 0.5156 |
| IsolationForest | late_night | 278 | 10 | 0.0360 |
| IsolationForest | ordinary_human | 226015 | 966 | 0.0043 |
| IsolationForest | service_account | 56283 | 0 | 0.0000 |
| IsolationForest | travel | 1582 | 29 | 0.0183 |
| LOF | forgot_password | 289 | 10 | 0.0346 |
| LOF | late_night | 278 | 115 | 0.4137 |
| LOF | ordinary_human | 226015 | 2774 | 0.0123 |
| LOF | service_account | 56283 | 822 | 0.0146 |
| LOF | travel | 1582 | 50 | 0.0316 |
| Autoencoder | forgot_password | 289 | 101 | 0.3495 |
| Autoencoder | late_night | 278 | 41 | 0.1475 |
| Autoencoder | ordinary_human | 226015 | 2744 | 0.0121 |
| Autoencoder | service_account | 56283 | 0 | 0.0000 |
| Autoencoder | travel | 1582 | 28 | 0.0177 |
| Ensemble-mean | forgot_password | 289 | 62 | 0.2145 |
| Ensemble-mean | late_night | 278 | 93 | 0.3345 |
| Ensemble-mean | ordinary_human | 226015 | 1886 | 0.0083 |
| Ensemble-mean | service_account | 56283 | 0 | 0.0000 |
| Ensemble-mean | travel | 1582 | 56 | 0.0354 |
| Ensemble-max | forgot_password | 289 | 69 | 0.2388 |
| Ensemble-max | late_night | 278 | 74 | 0.2662 |
| Ensemble-max | ordinary_human | 226015 | 1787 | 0.0079 |
| Ensemble-max | service_account | 56283 | 305 | 0.0054 |
| Ensemble-max | travel | 1582 | 11 | 0.0070 |
| Rules | forgot_password | 289 | 108 | 0.3737 |
| Rules | late_night | 278 | 1 | 0.0036 |
| Rules | ordinary_human | 226015 | 1892 | 0.0084 |
| Rules | service_account | 56283 | 0 | 0.0000 |
| Rules | travel | 1582 | 57 | 0.0360 |

## Feature-group ablation (Isolation Forest)

| removed_group | roc_auc_mean | roc_auc_std | avg_precision_mean | avg_precision_std | f1_mean | f1_std |
|---|---|---|---|---|---|---|
| none | 0.978 | 0.009 | 0.737 | 0.102 | 0.655 | 0.102 |
| event | 0.983 | 0.008 | 0.758 | 0.085 | 0.651 | 0.090 |
| time | 0.936 | 0.023 | 0.672 | 0.127 | 0.640 | 0.098 |
| ip_window | 0.977 | 0.010 | 0.747 | 0.120 | 0.680 | 0.102 |
| user_window | 0.974 | 0.009 | 0.727 | 0.097 | 0.648 | 0.094 |
| profile | 0.953 | 0.021 | 0.672 | 0.129 | 0.620 | 0.096 |

## Alert-budget sweep (mean over seeds)

| method | budget | precision | recall | f1 | alert_rate |
|---|---|---|---|---|---|
| IsolationForest | 0.003 | 0.800 | 0.067 | 0.119 | 0.002 |
| IsolationForest | 0.005 | 0.800 | 0.140 | 0.227 | 0.003 |
| IsolationForest | 0.010 | 0.971 | 0.361 | 0.486 | 0.008 |
| IsolationForest | 0.020 | 0.775 | 0.594 | 0.655 | 0.016 |
| IsolationForest | 0.050 | 0.363 | 0.809 | 0.500 | 0.044 |
| IsolationForest | 0.100 | 0.190 | 0.912 | 0.314 | 0.093 |
| LOF | 0.003 | 0.321 | 0.032 | 0.058 | 0.002 |
| LOF | 0.005 | 0.316 | 0.052 | 0.088 | 0.004 |
| LOF | 0.010 | 0.257 | 0.078 | 0.115 | 0.007 |
| LOF | 0.020 | 0.209 | 0.184 | 0.192 | 0.017 |
| LOF | 0.050 | 0.155 | 0.369 | 0.217 | 0.044 |
| LOF | 0.100 | 0.106 | 0.514 | 0.175 | 0.091 |
| Autoencoder | 0.003 | 0.609 | 0.060 | 0.100 | 0.002 |
| Autoencoder | 0.005 | 0.429 | 0.110 | 0.166 | 0.005 |
| Autoencoder | 0.010 | 0.455 | 0.194 | 0.255 | 0.009 |
| Autoencoder | 0.020 | 0.387 | 0.323 | 0.344 | 0.017 |
| Autoencoder | 0.050 | 0.304 | 0.703 | 0.423 | 0.045 |
| Autoencoder | 0.100 | 0.171 | 0.847 | 0.283 | 0.097 |
| Ensemble-mean | 0.003 | 0.675 | 0.088 | 0.147 | 0.002 |
| Ensemble-mean | 0.005 | 0.819 | 0.130 | 0.202 | 0.004 |
| Ensemble-mean | 0.010 | 0.729 | 0.197 | 0.276 | 0.007 |
| Ensemble-mean | 0.020 | 0.534 | 0.404 | 0.448 | 0.015 |
| Ensemble-mean | 0.050 | 0.301 | 0.706 | 0.421 | 0.045 |
| Ensemble-mean | 0.100 | 0.171 | 0.838 | 0.283 | 0.094 |
| Ensemble-max | 0.003 | 0.461 | 0.055 | 0.095 | 0.002 |
| Ensemble-max | 0.005 | 0.480 | 0.099 | 0.159 | 0.004 |
| Ensemble-max | 0.010 | 0.501 | 0.209 | 0.287 | 0.008 |
| Ensemble-max | 0.020 | 0.541 | 0.466 | 0.487 | 0.017 |
| Ensemble-max | 0.050 | 0.324 | 0.725 | 0.445 | 0.044 |
| Ensemble-max | 0.100 | 0.190 | 0.906 | 0.313 | 0.093 |

## LOF neighbourhood-size sweep (mean over seeds)

| k | roc_auc | avg_precision | f1 | auc_brute_force | auc_odd_hour_login | auc_password_spray | auc_privilege_escalation | auc_rare_ip_takeover |
|---|---|---|---|---|---|---|---|---|
| 10.000 | 0.772 | 0.317 | 0.380 | 0.980 | 0.746 | 0.761 | 0.420 | 0.704 |
| 35.000 | 0.810 | 0.141 | 0.192 | 0.844 | 0.916 | 0.730 | 0.617 | 0.831 |
| 100.000 | 0.788 | 0.203 | 0.263 | 0.602 | 0.989 | 0.593 | 0.959 | 0.923 |
| 250.000 | 0.881 | 0.240 | 0.294 | 0.801 | 0.991 | 0.723 | 0.959 | 0.950 |

Total runtime: 549 s
