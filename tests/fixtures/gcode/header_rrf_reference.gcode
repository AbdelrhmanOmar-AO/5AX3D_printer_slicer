G21 ; set units to millimeters
G90 ; use absolute coordinates
M190 S55 ; wait for bed temperature to be reached
G32 ; homing and bed calibration
M104 S210 ; set temperature
M109 S210 ; wait for temperature to be reached
T0
M82 ; use absolute distances for extrusion
; switch to enable 3Z mode
M98 P"/macros/enable3Z.g"
M400 ; wait 
; purging line
G92 E0
G1 Z76.3 U76.3 V76.3 F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z75.3 U75.3 V75.3 F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z75.3 U75.3 V75.3 F1000.0 E15 ; draw 1st line
G1 X0.4 Y200.0 Z75.3 U75.3 V75.3 F1000.0 ; move to side a little
G1 X0.4 Y20 Z75.3 U75.3 V75.3 F1000.0 E30 ; draw 2nd line
G1 E28.0 F2700 ; retract
; done purging extruder
M83 ; relative extrusion
