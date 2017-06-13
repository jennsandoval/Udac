SELECT 
TOP 10 *

FROM 

(
SELECT value, 
COUNT(*) AS num 

FROM nodes_tags 

INNER JOIN
 
(
SELECT DISTINCT (id) 

FROM nodes_tags 

WHERE value = 'restaurant')  AS restaurant_type 

ON nodes_tags.id  = restaurant_type.id 


WHERE key = 'name' 

GROUP BY value 

ORDER BY COUNT(*) DESC);