from tctrack_class import TCTrack

tctrack_test = TCTrack(config_file="./experiments/TCTrack/config.yaml", snapshot="./tools/snapshot/checkpoint00_e84.pth", coordinates=[(1,100),(2,100)])
tctrack_test.run()
