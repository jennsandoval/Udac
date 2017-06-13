SELECT 
top 10 e.user, 
COUNT(*) as num


FROM

(
SELECT user 
FROM nodes 

UNION ALL 

SELECT user FROM ways
) e

WHERE e.user is not null

GROUP BY e.user

ORDER BY COUNT(*) DESC;