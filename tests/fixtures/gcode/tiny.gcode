; tiny fixture mimicking atom.kinematics3z output; exactly 20 lines
G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; use absolute distances for extrusion
G92 E0
G1 X0.1 Y20 Z75.3 U75.3 V75.3 F1000.0 ; move to start-line position
G1 X0.1 Y200.0 Z75.3 U75.3 V75.3 F1000.0 E15 ; draw 1st line
G1 E13.0 F2700 ; retract
M83 ; relative extrusion
G1 X10.0 Y10.0 Z75.5 U75.5 V75.5 E0.5 F600
G1 X11.0 Y10.0 Z75.5 U75.6 V75.4 E0.25 F600
G1 E-2.0 F2700 ; retract
G0 X20.0 Y20.0 Z78.0 U78.0 V78.0 F3000 ; travel
G1 E2.0 F2700 ; prime
G1 X21.0 Y20.0 Z78.0 U78.1 V77.9 E0.3 F600
M82 ; absolute extrusion
G92 E0
G1 E-2.0 F2700 ; retract
M104 S0 ; turn off temperature
M400 ; wait
