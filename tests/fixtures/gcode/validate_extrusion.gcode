; P1.4 validator fixture: 7.5 mm of filament in one move on line 14 (EXTRUSION)
G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; use absolute distances for extrusion
M98 P"/macros/enable3Z.g"
G92 E0
G1 Z76.3 U76.3 V76.3 F500 ; move z up little to prevent scratching of surface
G1 X0.1 Y20 Z75.3 U75.3 V75.3 F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z75.3 U75.3 V75.3 F1000.0 E15 ; draw 1st line
G1 E13.0 F2700 ; retract
M83 ; relative extrusion
G1 X150.0 Y145.0 Z85.0 U85.0 V85.0 E0.000000 F3000
G1 E2.00 F2700 ; prime
G1 X151.0 Y145.0 Z85.0 U85.0 V85.0 E7.500000 F600
G1 X152.0 Y145.0 Z85.5 U84.5 V85.0 E0.080000 F608
G1 E-2.00 F2700 ; retract
G1 X160.0 Y150.0 Z90.0 U90.0 V90.0 E0.000000 F3000
G1 E2.00 F2700 ; prime
G1 X161.0 Y150.0 Z90.0 U90.0 V90.0 E0.080000 F600
M82 ; absolute extrusion
G92 E0
G1 E-2.0 F2700 ; retract
M104 S0 ; turn off temperature
M98 P"/macros/disable3Z.g"
M400 ; wait
