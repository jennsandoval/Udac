SELECT 
TOP 10 amenity_type.value, 
COUNT(*) AS num

FROM 

(
SELECT value 

FROM
nodes_tags

WHERE
key = 'amenity')  AS amenity_type

GROUP BY amenity_type.value

ORDER BY COUNT(*) DESC;