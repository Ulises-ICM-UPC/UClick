#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import os
import sys
#
sys.path.insert(0, 'uclick')
import uclick as uclick
#
pathFolderMain = 'example'
assert os.path.exists(pathFolderMain)
#
#''' --------------------------------------------------------------------------
# Click horizon
#''' --------------------------------------------------------------------------
#
pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
overwrite = True # USER DEFINED
verbosePlot = True # USER DEFINED
#
uclick.ClickHorizon(pathFolderBasis, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Click GCPs
#''' --------------------------------------------------------------------------
#
#pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
overwrite = False # USER DEFINED
verbosePlot = True # USER DEFINED
#
uclick.ClickGCPs(pathFolderMain, pathFolderBasis, overwrite, verbosePlot)
#
