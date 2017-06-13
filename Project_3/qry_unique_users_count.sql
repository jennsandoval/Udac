SELECT 
COUNT(unique_users.user_id)

FROM

(
SELECT 
DISTINCT users.uid as user_id

FROM 

(
SELECT uid
FROM nodes

UNION ALL

SELECT uid
FROM ways
) users

) unique_users;