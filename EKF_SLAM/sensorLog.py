from frame2d import Frame2D
robotFrames = [
   (0, Frame2D.fromXYA(0.000000,0.000000,0.000136)),
   (1, Frame2D.fromXYA(0.000000,0.000000,0.000134)),
   (2, Frame2D.fromXYA(0.000000,0.000000,0.000098)),
   (3, Frame2D.fromXYA(0.000000,0.000000,0.000090)),
   (4, Frame2D.fromXYA(0.000000,0.000000,0.000128)),
   (5, Frame2D.fromXYA(0.000000,0.000000,0.000184)),
   (6, Frame2D.fromXYA(0.000000,0.000000,0.000172)),
   (7, Frame2D.fromXYA(0.000000,0.000000,0.000149)),
   (8, Frame2D.fromXYA(0.000000,0.000000,0.000200)),
   (9, Frame2D.fromXYA(0.000000,0.000000,0.000205)),
   (10, Frame2D.fromXYA(0.000000,0.000000,0.000241)),
   (11, Frame2D.fromXYA(0.000000,0.000000,0.000255)),
   (12, Frame2D.fromXYA(0.000000,0.000000,0.000263)),
   (13, Frame2D.fromXYA(0.000000,0.000000,0.000292)),
   (14, Frame2D.fromXYA(0.000000,0.000000,0.000277)),
   (15, Frame2D.fromXYA(0.000000,0.000000,0.000253)),
   (16, Frame2D.fromXYA(0.000000,0.000000,0.000341)),
   (17, Frame2D.fromXYA(0.000000,0.000000,0.000356)),
   (18, Frame2D.fromXYA(0.000000,0.000000,0.000343)),
   (19, Frame2D.fromXYA(0.000000,0.000000,0.000346)),
   (20, Frame2D.fromXYA(0.000000,0.000000,0.000356)),
   (21, Frame2D.fromXYA(0.000000,0.000000,0.000340)),
   (22, Frame2D.fromXYA(-0.313148,0.101132,0.006018)),
   (23, Frame2D.fromXYA(-0.462175,0.865805,0.044445)),
   (24, Frame2D.fromXYA(-1.104813,4.580005,0.233520)),
   (25, Frame2D.fromXYA(-3.249830,9.724330,0.512946)),
   (26, Frame2D.fromXYA(-4.641544,11.961404,0.655750)),
   (27, Frame2D.fromXYA(-4.764666,12.128667,0.678113)),
   (28, Frame2D.fromXYA(-4.764666,12.128667,0.678123)),
   (29, Frame2D.fromXYA(-4.764666,12.128667,0.678145)),
   (30, Frame2D.fromXYA(-4.764666,12.128667,0.678190)),
   (31, Frame2D.fromXYA(-4.764666,12.128667,0.678153)),
   (32, Frame2D.fromXYA(-4.764666,12.128667,0.678111)),
   (33, Frame2D.fromXYA(60.302048,-28.504730,0.169737)),
   (34, Frame2D.fromXYA(60.304443,-28.500889,0.169794)),
   (35, Frame2D.fromXYA(60.304443,-28.500889,0.169800)),
   (36, Frame2D.fromXYA(60.304443,-28.500889,0.169885)),
   (37, Frame2D.fromXYA(60.304443,-28.500889,0.169861)),
   (38, Frame2D.fromXYA(60.304443,-28.500889,0.169848)),
   (39, Frame2D.fromXYA(60.304443,-28.500889,0.169871)),
   (40, Frame2D.fromXYA(60.304443,-28.500889,0.169849)),
   (41, Frame2D.fromXYA(60.304443,-28.500889,0.169815)),
   (42, Frame2D.fromXYA(60.304443,-28.500889,0.169756)),
   (43, Frame2D.fromXYA(60.304443,-28.500889,0.169732)),
   (44, Frame2D.fromXYA(60.117012,-28.536392,0.169583)),
   (45, Frame2D.fromXYA(56.743950,-29.277285,0.153763)),
   (46, Frame2D.fromXYA(50.870327,-30.287266,0.144845)),
   (47, Frame2D.fromXYA(44.369133,-31.338688,0.132657)),
   (48, Frame2D.fromXYA(40.091507,-31.930117,0.129288)),
   (49, Frame2D.fromXYA(36.175762,-32.465153,0.126265)),
   (50, Frame2D.fromXYA(36.175762,-32.465153,0.126610)),
   (51, Frame2D.fromXYA(36.175762,-32.465153,0.126638)),
   (52, Frame2D.fromXYA(36.175762,-32.465153,0.126658)),
   (53, Frame2D.fromXYA(37.606625,-34.834282,0.138702)),
   (54, Frame2D.fromXYA(37.607903,-34.832909,0.138748)),
   (55, Frame2D.fromXYA(37.607903,-34.832909,0.138646)),
   (56, Frame2D.fromXYA(37.607903,-34.832909,0.138645)),
   (57, Frame2D.fromXYA(37.607903,-34.832909,0.138605)),
   (58, Frame2D.fromXYA(37.607903,-34.832909,0.138635)),
   (59, Frame2D.fromXYA(37.607903,-34.832909,0.138571)),
   (60, Frame2D.fromXYA(37.976921,-34.772446,0.139141)),
   (61, Frame2D.fromXYA(37.349945,-32.067879,0.277998)),
   (62, Frame2D.fromXYA(37.349945,-32.067879,0.277998)),
   (63, Frame2D.fromXYA(34.184753,-26.291878,0.609263)),
   (64, Frame2D.fromXYA(33.835350,-25.809132,0.657249)),
   (65, Frame2D.fromXYA(33.835350,-25.809132,0.657214)),
   (66, Frame2D.fromXYA(33.835350,-25.809132,0.657189)),
   (67, Frame2D.fromXYA(33.835350,-25.809132,0.657229)),
   (68, Frame2D.fromXYA(33.835350,-25.809132,0.657281)),
   (69, Frame2D.fromXYA(33.835350,-25.809132,0.657268)),
   (70, Frame2D.fromXYA(35.669880,-24.603781,0.647672)),
   (71, Frame2D.fromXYA(36.777229,-23.778675,0.646858)),
   (72, Frame2D.fromXYA(45.178226,-17.534023,0.642381)),
   (73, Frame2D.fromXYA(49.095459,-14.606279,0.641941)),
   (74, Frame2D.fromXYA(54.520481,-10.548215,0.643330)),
   (75, Frame2D.fromXYA(54.588169,-10.435135,0.649148)),
   (76, Frame2D.fromXYA(54.588169,-10.435135,0.649166)),
   (77, Frame2D.fromXYA(54.588169,-10.435135,0.649167)),
   (78, Frame2D.fromXYA(54.588169,-10.435135,0.649206)),
   (79, Frame2D.fromXYA(54.588169,-10.435135,0.649200)),
   (80, Frame2D.fromXYA(54.588169,-10.435135,0.649230)),
   (81, Frame2D.fromXYA(54.588169,-10.435135,0.649222)),
   (82, Frame2D.fromXYA(54.588169,-10.435135,0.649135)),
   (83, Frame2D.fromXYA(54.588169,-10.435135,0.680643)),
   (84, Frame2D.fromXYA(54.588169,-10.435135,0.715699)),
   (85, Frame2D.fromXYA(54.588169,-10.435135,0.771619)),
   (86, Frame2D.fromXYA(0.000000,0.000000,0.000000)),
   (87, Frame2D.fromXYA(0.000000,0.000000,0.037339)),
   (88, Frame2D.fromXYA(0.000000,0.000000,0.251872)),
   (89, Frame2D.fromXYA(0.000000,0.000000,0.309750)),
   (90, Frame2D.fromXYA(0.000000,0.000000,0.419830)),
   (91, Frame2D.fromXYA(0.000000,0.000000,0.424847)),
   (92, Frame2D.fromXYA(0.000000,0.000000,0.460969)),
   (93, Frame2D.fromXYA(0.000000,0.000000,0.439331)),
   (94, Frame2D.fromXYA(0.000000,0.000000,0.423976)),
   (95, Frame2D.fromXYA(0.000000,0.000000,0.418559)),
   (96, Frame2D.fromXYA(0.000000,0.000000,0.417142)),
   (97, Frame2D.fromXYA(0.000000,0.000000,0.414768)),
   (98, Frame2D.fromXYA(0.000000,0.000000,0.400724)),
   (99, Frame2D.fromXYA(0.000000,0.000000,0.387131))]
cubeFrames = [
   (28, Frame2D.fromXYA(117.225739,128.686523,0.553218)),
   (29, Frame2D.fromXYA(117.225739,128.686523,0.553218)),
   (30, Frame2D.fromXYA(117.225739,128.686523,0.553218)),
   (31, Frame2D.fromXYA(117.225739,128.686523,0.553218)),
   (32, Frame2D.fromXYA(117.308868,129.094788,0.564009)),
   (33, Frame2D.fromXYA(223.836304,14.179157,0.052367)),
   (34, Frame2D.fromXYA(223.840332,14.171091,0.053001)),
   (35, Frame2D.fromXYA(223.776611,14.074574,0.049064)),
   (36, Frame2D.fromXYA(223.797592,14.169754,0.052581)),
   (37, Frame2D.fromXYA(223.797592,14.169754,0.052581)),
   (38, Frame2D.fromXYA(223.905640,14.280319,0.055826)),
   (39, Frame2D.fromXYA(223.884659,14.231117,0.053664)),
   (40, Frame2D.fromXYA(223.727158,14.068264,0.048766)),
   (41, Frame2D.fromXYA(223.705826,14.003426,0.045684)),
   (42, Frame2D.fromXYA(223.758484,14.095760,0.049911)),
   (43, Frame2D.fromXYA(224.002975,14.343975,0.057914)),
   (44, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (45, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (46, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (47, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (48, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (49, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (50, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (51, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (52, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (53, Frame2D.fromXYA(223.971207,14.282055,0.055899)),
   (54, Frame2D.fromXYA(224.043640,14.414124,0.060693)),
   (55, Frame2D.fromXYA(224.085480,14.394577,0.059323)),
   (56, Frame2D.fromXYA(224.234406,14.525829,0.064086)),
   (57, Frame2D.fromXYA(223.917023,14.267281,0.056205)),
   (58, Frame2D.fromXYA(223.917023,14.267281,0.056205)),
   (59, Frame2D.fromXYA(224.003906,14.343830,0.059561)),
   (60, Frame2D.fromXYA(224.033203,14.305901,0.057494)),
   (61, Frame2D.fromXYA(224.033203,14.305901,0.057494)),
   (62, Frame2D.fromXYA(224.033203,14.305901,0.057494)),
   (63, Frame2D.fromXYA(224.033203,14.305901,0.057494)),
   (64, Frame2D.fromXYA(224.033203,14.305901,0.057494))]
cliffSensor = [
   (0, False),
   (1, False),
   (2, False),
   (3, False),
   (4, False),
   (5, False),
   (6, False),
   (7, False),
   (8, False),
   (9, False),
   (10, False),
   (11, False),
   (12, False),
   (13, False),
   (14, False),
   (15, False),
   (16, False),
   (17, False),
   (18, False),
   (19, False),
   (20, False),
   (21, False),
   (22, False),
   (23, False),
   (24, False),
   (25, False),
   (26, False),
   (27, False),
   (28, False),
   (29, False),
   (30, False),
   (31, False),
   (32, False),
   (33, False),
   (34, False),
   (35, False),
   (36, False),
   (37, False),
   (38, False),
   (39, False),
   (40, False),
   (41, False),
   (42, False),
   (43, False),
   (44, False),
   (45, False),
   (46, False),
   (47, False),
   (48, False),
   (49, False),
   (50, False),
   (51, False),
   (52, False),
   (53, False),
   (54, False),
   (55, False),
   (56, False),
   (57, False),
   (58, False),
   (59, False),
   (60, False),
   (61, False),
   (62, False),
   (63, False),
   (64, False),
   (65, False),
   (66, False),
   (67, False),
   (68, False),
   (69, False),
   (70, False),
   (71, False),
   (72, False),
   (73, False),
   (74, False),
   (75, False),
   (76, False),
   (77, False),
   (78, False),
   (79, False),
   (80, False),
   (81, False),
   (82, False),
   (83, False),
   (84, False),
   (85, True),
   (86, True),
   (87, True),
   (88, True),
   (89, True),
   (90, True),
   (91, True),
   (92, True),
   (93, True),
   (94, True),
   (95, True),
   (96, True),
   (97, True),
   (98, True),
   (99, True)]