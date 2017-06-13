SELECT
cityname.value,
COUNT(*) as num

FROM

(SELECT
value


FROM
nodes_tags


WHERE
key = 'city') cityname

GROUP BY
cityname.value

ORDER BY
COUNT(*) DESC;