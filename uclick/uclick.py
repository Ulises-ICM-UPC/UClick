#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import cv2
import os
import sys
#
import ulises_uclick as uclick
#
def ClickHorizon(pathBasis, overwrite, verbosePlot):
    #
    for root, _, filenames in sorted(os.walk(pathBasis)):
        for filename in sorted([item for item in filenames if item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']]):
            #
            print('... clicking horizon at {:}'.format(filename))
            if input('Continue (n/[y])? ') in ['n', 'N']:
                print('stop clicking horizon') 
                sys.exit()
            #
            img = cv2.imread(root + os.sep + filename)
            #
            pathCdhTxt = root + os.sep + filename[0:filename.rfind('.')] + 'cdh.txt'
            uclick.ClickCdhTxt(img, pathCdhTxt, options={'overwrite':overwrite, 'titlePreamble':'{:}: '.format(filename)})
            #
            if verbosePlot:
                chs, rhs = uclick.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
                if len(chs) == len(rhs) == 0:
                    continue
                pathFolderOut = pathBasis + os.sep + '..' + os.sep + 'TMP' # IMP*
                uclick.MakeFolder(pathFolderOut)
                uclick.PlotCRWithTextsInImage(img, chs, rhs, pathFolderOut + os.sep + filename[0:-4] + 'cdh_check.jpg', options={'texts':None})
    #
    return None
#
def ClickGCPs(pathMain, pathBasis, overwrite, verbosePlot):
    for root, _, filenames in sorted(os.walk(pathBasis)):
        for filename in sorted([item for item in filenames if item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']]):
            #
            print('... clicking GCPs at {:}'.format(filename))
            if input('Continue (n/[y])? ') in ['n', 'N']:
                print('stop clicking GCPs') 
                sys.exit()
            #
            img = cv2.imread(root + os.sep + filename)
            #
            pathGCPsTxt = pathMain + os.sep + 'GCPs.txt'
            pathCdgTxt = root + os.sep + filename[0:filename.rfind('.')] + 'cdg.txt'
            uclick.ClickCdgTxt(img, pathGCPsTxt, pathCdgTxt, options={'overwrite':overwrite, 'titlePreamble':'{:}: '.format(filename)})
            #
            if verbosePlot:
                cs, rs, _, _, _, codes = uclick.ReadCdgTxt(pathCdgTxt, options={'readCodes':True, 'readOnlyGood':True})
                if len(cs) == len(rs) == 0:
                    continue
                pathFolderOut = pathBasis + os.sep + '..' + os.sep + 'TMP' # IMP*
                uclick.MakeFolder(pathFolderOut)
                uclick.PlotCRWithTextsInImage(img, cs, rs, pathFolderOut + os.sep + filename[0:-4] + 'cdg_check.jpg', options={'texts':codes})
    return None
