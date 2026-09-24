\# Monitoring \& Alerting Decisions



\## Metrics we expose

\- Request count

\- Error count \& error rate

\- Average latency

\- Prediction distribution (on\_time vs late)



\## Prediction logs

Every prediction is stored in `logs/predictions.jsonl` with:

\- Timestamp

\- Full input order

\- Prediction, probability, label

\- Model version

\- Latency



This allows us to evaluate model performance later when the real delivery date arrives.



\## What we would alert on



| Alert | Condition | Why |

|-------|-----------|-----|

| High Error Rate | error\_rate > 5% over the last 10 minutes | Indicates problems with input data or model loading |

| High Latency | avg\_latency > 10 seconds | Service is too slow for production use |

| Sudden shift in predictions | % of "late" predictions changes by more than 20% compared to the last 24 hours | Possible data drift or concept drift |

| No traffic | Zero requests for more than 1 hour during business hours | Service might be down or not receiving traffic |



\## Future improvements

\- Send alerts to Slack / email

\- Use Prometheus + Grafana for better visualization

\- Add data drift detection (e.g. with Evidently or Alibi Detect)
