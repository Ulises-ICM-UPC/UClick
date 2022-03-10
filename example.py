#'''
# Created on 2022 by Gonzalo Simarro and Daniel Calvete
#'''
#
import os
import sys
#
sys.path.insert(0, 'uclick')
import uclick as uclick
#
pathFolderMain = 'example' # USER DEFINED
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
#''' --------------------------------------------------------------------------
# check GCPs
#''' --------------------------------------------------------------------------
#
#pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
eCritical = 5. # USER DEFINED (eCritical is in pixels)
#
print('Checking of the GCPs')
uclick.CheckGCPs(pathFolderBasis, eCritical)
#
