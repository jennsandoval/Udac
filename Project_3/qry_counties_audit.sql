SELECT 
countyname.value, 

countyname.key,

count(*) AS num


FROM 
(SELECT value, key 

FROM ways_tags UNION ALL

SELECT value, key
FROM nodes_tags
)  AS countyname


WHERE countyname.key = 'county'

GROUP BY countyname.value, countyname.key


ORDER BY count(*) DESC;