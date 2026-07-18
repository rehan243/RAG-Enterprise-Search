CREATE OR REPLACE VIEW fraud_analysis AS
SELECT 
    transactions.id AS transaction_id,
    transactions.amount AS transaction_amount,
    users.id AS user_id,
    users.name AS user_name,
    users.email AS user_email,
    transactions.created_at AS transaction_date,
    CASE 
        WHEN transactions.amount > 1000 THEN 'high_value'
        WHEN transactions.amount BETWEEN 500 AND 1000 THEN 'medium_value'
        ELSE 'low_value'
    END AS value_category,
    COUNT(transactions.id) OVER (PARTITION BY users.id) AS transaction_count,
    SUM(transactions.amount) OVER (PARTITION BY users.id) AS total_spent
FROM 
    transactions
JOIN 
    users ON transactions.user_id = users.id
WHERE 
    transactions.status = 'completed'
    AND transactions.created_at >= NOW() - INTERVAL '1 year'
ORDER BY 
    transaction_date DESC;

-- TODO: consider adding more filters for specific user segments
-- maybe include geo-location or device type later