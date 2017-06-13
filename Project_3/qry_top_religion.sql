SELECT 
religion_type.value, 
COUNT(*) AS num

FROM 

(
SELECT value 

FROM
nodes_tags

WHERE
key = 'religion')  AS religion_type


GROUP BY religion_type.value

ORDER BY COUNT(*) DESC;