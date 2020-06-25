<!DOCTYPE html>
<html>
<head>
	<title>sudoku solver</title>
</head>
<body>
<?php 

$command = escapeshellcmd('solver.py');
$output = shell_exec($command);


?>
</body>
</html>