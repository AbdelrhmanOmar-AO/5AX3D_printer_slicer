M82 ; absolute extrusion
G92 E0
G1 E-2.0 F2700 ; retract
G92 E0
M104 S0 ; turn off temperature
M140 S0
M106 S0    ; fan off
; switch to disable 3Z mode
M98 P"/macros/disable3Z.g"
M400 ; wait 
