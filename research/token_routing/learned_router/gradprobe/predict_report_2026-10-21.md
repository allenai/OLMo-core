| ckpt | task | rule | trained outcome | keep | grad cos | log10 norm ratio | mean dCE | leak: count AUC-0.5 | leak: whole AUC-0.5 | rows |
|---|---|---|---|---|---|---|---|---|---|---|
| per-task | contradiction | a0.5_abs | ? | 0.39 | 0.53 | +0.42 | +0.075 | +0.44 | +0.12 | 24 |
| per-task | contradiction | rand | ? | 0.39 | 0.09 | +3.20 | +1.658 | +0.15 | +0.00 | 24 |
| per-task | contradiction | hfl | ? | 0.36 | 0.07 | +3.14 | +0.735 | +0.12 | +0.00 | 24 |
| per-task | contradiction | hdr17 | L | 0.32 | 0.66 | +0.10 | +0.003 | +0.43 | +0.42 | 24 |
| per-task | contradiction | hdr8 | L | 0.25 | 0.62 | +0.11 | +0.001 | +0.46 | +0.46 | 24 |
| per-task | contradiction | hdr3 | L | 0.21 | 0.60 | +0.10 | +0.006 | +0.49 | +0.49 | 24 |
| per-task | contradiction | goldhfl | L | 0.39 | 0.57 | +0.12 | +0.002 | +0.49 | +0.50 | 24 |
| per-task | contradiction | ck64 | W | 0.96 | 0.98 | +0.05 | +0.001 | +0.12 | +0.03 | 24 |
| per-task | contradiction | ck32 | W | 0.79 | 0.68 | +0.30 | +0.005 | +0.05 | -0.09 | 24 |
| per-task | contradiction | hdr33 | W | 0.45 | 0.63 | +0.12 | +0.001 | +0.37 | +0.33 | 24 |
| per-task | contradiction | ck16 | W | 0.53 | 0.09 | +2.53 | +0.258 | +0.06 | -0.06 | 24 |
| per-task | nq | a0.5_abs | ? | 0.26 | 0.98 | +0.05 | +0.001 | +0.49 | +0.00 | 24 |
| per-task | nq | hfl | ? | 0.24 | 0.49 | +0.90 | +0.309 | -0.10 | +0.00 | 24 |
| per-task | nq | rand | ? | 0.26 | 0.24 | +1.86 | +1.130 | -0.13 | +0.00 | 24 |
| per-task | nq | goldhfl | L | 0.26 | 0.76 | -0.70 | -0.027 | +0.50 | +0.50 | 24 |
| per-task | nq | ck64 | L | 0.41 | 0.61 | +0.52 | +0.360 | +0.00 | +0.00 | 24 |
| per-task | nq | ck32 | L | 0.21 | 0.46 | +1.03 | +0.296 | +0.00 | +0.00 | 24 |
| per-task | nq | kv17 | L | 0.19 | 0.43 | -0.08 | -0.022 | +0.38 | +0.41 | 24 |
| per-task | nq | kv33 | W | 0.35 | 0.68 | -0.18 | -0.021 | +0.28 | +0.33 | 24 |
| per-task | outlier | a0.5_abs | ? | 0.29 | 0.49 | -0.06 | +0.025 | +0.47 | +0.00 | 24 |
| per-task | outlier | rand | ? | 0.29 | 0.05 | +3.17 | +1.222 | +0.08 | +0.00 | 24 |
| per-task | outlier | hfl | ? | 0.25 | 0.04 | +2.88 | +0.543 | +0.05 | +0.00 | 24 |
| per-task | outlier | goldhfl | L | 0.29 | 0.26 | +0.12 | -0.007 | +0.50 | +0.50 | 24 |
| per-task | outlier | kv33 | L | 0.37 | 0.14 | +1.88 | +0.127 | +0.35 | +0.33 | 24 |
| per-task | outlier | ck0 | L | 0.04 | 0.07 | +2.84 | +0.876 | +0.08 | +0.00 | 24 |
| per-task | outlier | ck16 | L | 0.15 | 0.05 | +3.06 | +0.673 | +0.08 | +0.00 | 24 |
| per-task | outlier | gb33 | L | 0.33 | 0.04 | +3.04 | +1.455 | -0.03 | -0.03 | 24 |
| per-task | outlier | ck64 | W | 0.49 | 0.23 | +1.04 | +0.134 | +0.08 | +0.00 | 24 |
| per-task | outlier | flk64 | W | 0.49 | 0.21 | +1.02 | +0.131 | +0.08 | +0.00 | 24 |
| setA32k | contradiction | a0.5_abs | ? | 0.39 | 0.27 | +0.77 | +0.059 | +0.45 | +0.08 | 24 |
| setA32k | contradiction | hfl | ? | 0.36 | 0.03 | +2.52 | +0.594 | +0.12 | +0.00 | 24 |
| setA32k | contradiction | rand | ? | 0.39 | 0.01 | +2.70 | +1.523 | +0.15 | +0.00 | 24 |
| setA32k | contradiction | hdr17 | L | 0.32 | 0.38 | +0.27 | -0.003 | +0.43 | +0.42 | 24 |
| setA32k | contradiction | goldhfl | L | 0.39 | 0.35 | +0.18 | +0.009 | +0.49 | +0.50 | 24 |
| setA32k | contradiction | hdr8 | L | 0.25 | 0.35 | +0.48 | +0.006 | +0.46 | +0.46 | 24 |
| setA32k | contradiction | hdr3 | L | 0.21 | 0.26 | +0.35 | +0.007 | +0.49 | +0.49 | 24 |
| setA32k | contradiction | ck64 | W | 0.96 | 0.93 | +0.04 | +0.001 | +0.12 | +0.03 | 24 |
| setA32k | contradiction | ck32 | W | 0.79 | 0.52 | +0.77 | +0.011 | +0.05 | -0.09 | 24 |
| setA32k | contradiction | hdr33 | W | 0.45 | 0.38 | +0.12 | +0.007 | +0.37 | +0.33 | 24 |
| setA32k | contradiction | ck16 | W | 0.53 | 0.10 | +1.66 | +0.185 | +0.06 | -0.06 | 24 |
| setA32k | nq | a0.5_abs | ? | 0.26 | 0.82 | -0.00 | -0.002 | +0.49 | +0.00 | 24 |
| setA32k | nq | hfl | ? | 0.24 | 0.28 | +1.67 | +0.412 | -0.10 | +0.00 | 24 |
| setA32k | nq | rand | ? | 0.26 | 0.04 | +3.05 | +2.188 | -0.13 | +0.00 | 24 |
| setA32k | nq | goldhfl | L | 0.26 | 0.61 | -0.92 | -0.032 | +0.50 | +0.50 | 24 |
| setA32k | nq | ck64 | L | 0.41 | 0.42 | +0.90 | +0.496 | +0.00 | +0.00 | 24 |
| setA32k | nq | kv17 | L | 0.19 | 0.36 | -0.66 | +0.201 | +0.38 | +0.41 | 24 |
| setA32k | nq | ck32 | L | 0.21 | 0.28 | +1.73 | +0.394 | +0.00 | +0.00 | 24 |
| setA32k | nq | kv33 | W | 0.35 | 0.66 | -0.31 | +0.040 | +0.28 | +0.33 | 24 |
| setA32k | outlier | a0.5_abs | ? | 0.29 | 0.34 | +0.55 | +0.002 | +0.48 | +0.00 | 24 |
| setA32k | outlier | hfl | ? | 0.25 | 0.12 | +2.47 | +0.116 | +0.05 | +0.00 | 24 |
| setA32k | outlier | rand | ? | 0.29 | 0.03 | +3.18 | +1.033 | +0.08 | +0.00 | 24 |
| setA32k | outlier | goldhfl | L | 0.29 | 0.31 | +0.65 | +0.011 | +0.50 | +0.50 | 24 |
| setA32k | outlier | kv33 | L | 0.37 | 0.17 | +2.70 | +0.204 | +0.35 | +0.33 | 24 |
| setA32k | outlier | ck0 | L | 0.04 | 0.10 | +2.90 | +0.717 | +0.08 | +0.00 | 24 |
| setA32k | outlier | ck16 | L | 0.15 | 0.10 | +3.07 | +0.304 | +0.08 | +0.00 | 24 |
| setA32k | outlier | gb33 | L | 0.33 | 0.07 | +3.25 | +1.253 | -0.03 | -0.03 | 24 |
| setA32k | outlier | ck64 | W | 0.49 | 0.32 | +1.14 | +0.020 | +0.08 | +0.00 | 24 |
| setA32k | outlier | flk64 | W | 0.49 | 0.29 | +0.97 | +0.012 | +0.08 | +0.00 | 24 |

AUC separating trained winners (W) from losers (L); 0.5 = no signal, 1.0 = perfect:

| metric (oriented) | per-task ckpts | setA32k | pooled |
|---|---|---|---|
| grad cos (higher = predicted good) | 0.65 (7W/13L) | 0.70 (7W/13L) | 0.64 (14W/26L) |
| -|log norm ratio| (closer to 1x = good) | 0.52 (7W/13L) | 0.65 (7W/13L) | 0.59 (14W/26L) |
| -dCE (dev-loss parity = good) | 0.58 (7W/13L) | 0.63 (7W/13L) | 0.60 (14W/26L) |
| -leak count (less visible-length leak = good) | 0.67 (7W/13L) | 0.67 (7W/13L) | 0.67 (14W/26L) |
| -leak whole (fewer gold-only whole docs = good) | 0.71 (7W/13L) | 0.71 (7W/13L) | 0.71 (14W/26L) |
