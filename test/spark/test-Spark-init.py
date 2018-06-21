import arimo.backend


arimo.backend.initSpark(
    sparkApp='test',
    yarnUpdateJARs=True,
    dataIO=arimo.backend.DATA_IO_OPTIONS)
