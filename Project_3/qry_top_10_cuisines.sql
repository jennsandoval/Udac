SELECT
TOP 10 cuisine_type.value,
COUNT(*) as num

FROM

(SELECT value 

FROM
nodes_tags

WHERE
key = 'cuisine') cuisine_type

GROUP BY
cuisine_type.value

ORDER BY COUNT(*) DESC