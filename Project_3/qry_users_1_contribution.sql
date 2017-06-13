SELECT 
COUNT(*)

FROM

(SELECT userlist.user, 
COUNT(*) as Contribution_Amount

FROM

(SELECT user 
FROM nodes 

UNION ALL 

SELECT user FROM ways) userlist


GROUP BY
userlist.user) User_Count


WHERE
User_Count.Contribution_Amount = 1;