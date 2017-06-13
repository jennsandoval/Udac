SELECT 
TOP 5 *

FROM 

(
SELECT value, 
COUNT(*) AS num 

FROM nodes_tags 

INNER JOIN
 
(
SELECT DISTINCT (id) 

FROM nodes_tags 

WHERE value = 'fast_food')  AS restaurant_type 

ON nodes_tags.id  = restaurant_type.id 


WHERE key = 'name' 

GROUP BY value 

ORDER BY COUNT(*) DESC);