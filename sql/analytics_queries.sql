-- dashboards for search quality; keep queries boring and fast on big fact tables

-- zero result rate by day: spikes usually mean tokenizer or acl regression
SELECT date_trunc('day', searched_at) AS day,
       count(*) FILTER (WHERE result_count = 0)::float / nullif(count(*), 0) AS zero_rate
FROM search_events
GROUP BY 1
ORDER BY 1 DESC
LIMIT 30;

-- latency percentiles by tenant because slos differ
SELECT tenant_id,
       percentile_disc(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       percentile_disc(0.9) WITHIN GROUP (ORDER BY latency_ms) AS p90_ms,
       percentile_disc(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
FROM search_events
WHERE searched_at > now() - interval '7 days'
GROUP BY tenant_id
ORDER BY p90_ms DESC;

-- click position distribution: if median moves right, ranking got worse
SELECT clicked_rank,
       count(*) AS clicks
FROM search_clicks
WHERE clicked_at > now() - interval '14 days'
GROUP BY clicked_rank
ORDER BY clicked_rank;

-- queries that never get clicks but have results: good candidates for human eval
SELECT q.query_text,
       count(*) AS impressions,
       count(c.click_id) AS clicks
FROM search_queries q
LEFT JOIN search_clicks c ON c.query_id = q.id
WHERE q.created_at > now() - interval '7 days'
GROUP BY q.query_text
HAVING count(*) > 50 AND count(c.click_id) = 0
ORDER BY impressions DESC
LIMIT 100;

-- embedding version drift: compare avg latency before and after rollout
SELECT embedding_model_version,
       avg(latency_ms) AS avg_ms,
       count(*) AS n
FROM search_events
WHERE searched_at > now() - interval '30 days'
GROUP BY embedding_model_version;

-- abandoned searches: user typed then left; often means slow ui or bad suggestions
SELECT date_trunc('hour', searched_at) AS hr,
       count(*) FILTER (WHERE clicked_at IS NULL)::float / nullif(count(*), 0) AS abandon_rate
FROM search_events
WHERE searched_at > now() - interval '3 days'
GROUP BY 1
ORDER BY 1 DESC;
